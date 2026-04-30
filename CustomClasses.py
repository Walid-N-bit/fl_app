import os
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch.nn.functional as F
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx
from collections.abc import Iterable
from flwr.app import Message, MetricRecord, RecordDict

import time, io
from collections.abc import Callable
from typing import Literal

from logging import INFO

from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, log, MessageType
from flwr.server import Grid

from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from custom_aggregation import perform_custom_aggregation


# class CustomStrat(FedAvg):
class CustomStrat(FedProx):
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
        use_custom_agg: int = 0,
    ) -> tuple[dict[int, Iterable[Message]], dict[int, Iterable[Message]], Result]:
        """
        Override start() method to return both replies and the result
        """

        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        evaluate_replies
            Object containing loss function, accuracy, and number of samples of clients per round
        
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        ########################################
        ########################################
        # added this to capture client metrics #
        clients_train_metrics = {}
        clients_eval_metrics = {}  # currently useless (eval happens in train())
        ########################################
        ########################################

        # for current_round in range(0, num_rounds + 1):
        for current_round in range(1, num_rounds + 1):

            ################################################
            ################################################

            # adding current-round to train_config for logging
            train_config.update({"current-round": current_round})
            ################################################
            ################################################

            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies

            # START: Round Time Measurement
            print(f"\n------- SENDING -------\n")

            round_start_time = time.time()

            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )
            print(f"\n------- RECEIVED -------\n")
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            #######################################################
            #######################################################
            # using custom aggregation method
            if use_custom_agg:
                agg_arrays = perform_custom_aggregation(train_replies, agg_arrays)
            #######################################################
            #######################################################

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            ###################################
            ###################################
            # saving client-side train metrics
            # clients_train_metrics[current_round] = self.compile_clients_metrics(
            #     train_replies
            # )
            current_round_client_metrics = self.compile_clients_metrics(train_replies)
            # same way you can add client-side eval metrics

            ##################################
            ##################################

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

            #################################################################
            #################################################################
            # learning-rate scheduler logic
            current_round_metrics = result.evaluate_metrics_serverapp.get(current_round)

            current_loss = current_round_metrics.get("loss")
            sch_patience = train_config.get("sch-patience")
            current_lr = train_config.get("classifier-lr")

            scheduler = CustomScheduler(current_lr, sch_patience, threshold=0.01)
            new_lr = scheduler.step(current_loss)

            train_config["classifier-lr"] = new_lr

            #################################################################
            #################################################################
            ###################################
            # 3. Inject Timing Data per Client
            ###################################
            if current_round_client_metrics:
                # print(f"\n{'-'*50}\n")
                # print(f"\n{current_round_client_metrics = }\n")

                for client_data in current_round_client_metrics:
                    # client_data is a dict containing merged MetricRecord and ConfigRecord

                    # Get the specific training time for this client (from MetricRecord)
                    # Note: Ensure your compile_clients_metrics merges the MetricRecord keys
                    client_train_time = client_data.get("train-time", 0)
                    # print(f"{client_data = }\n")
                    # print(f"{round_duration = }\n")
                    # print(f"{client_train_time = }\n")

                    # Calculate transmission time
                    # (Wall Clock Time) - (Time client actually spent working)
                    trans_time = round_duration - client_train_time
                    # print(f"{trans_time = }\n")

                    # Add new columns to the client data
                    client_data["round-time"] = round_duration
                    client_data["transmission-time"] = trans_time

            # Save to the main history dictionary
            clients_train_metrics[current_round] = current_round_client_metrics
            print(f"{clients_train_metrics = }\n")

            # 4. Add to Aggregated Metrics for Server CSV
            if agg_train_metrics is not None:
                agg_train_metrics["round-time"] = round_duration
            else:
                agg_train_metrics = MetricRecord({"round-time": round_duration})

            # print(f"{agg_train_metrics = }\n")

            result.train_metrics_clientapp[current_round] = agg_train_metrics
            print(f"\n{'-'*50}\n")

            #################################################################
            #################################################################

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return clients_train_metrics, clients_eval_metrics, result

    # added this method to prepare for building the global model by the server
    # server sends a prep flag to clients, clients reply ith their local classes
    def prepare(
        self,
        grid: Grid,
        arrays: ArrayRecord | None = None,
        prep_config: ConfigRecord | None = None,
        timeout: float = 3600,
    ):
        prep_replies = grid.send_and_receive(
            messages=self.configure_train(
                0,
                arrays,
                prep_config,
                grid,
            ),
            timeout=timeout,
        )

        return prep_replies

    # def compile_clients_metrics(self, replies: Iterable[Message]) -> list:
    #     """
    #     parse replies data and return a dict for metrics and the client that produced them

    #     :param replies: message replies from clients
    #     :type replies: Iterable[Message]
    #     :return: data to be returned to server
    #     :rtype: dict
    #     """
    #     round_metrics = []
    #     for msg in replies:
    #         if not msg.has_content():
    #             continue
    #         configs = msg.content["configs"]
    #         metrics = msg.content["metrics"]
    #         keys = metrics.keys()
    #         item = {
    #             "client-name": configs.get("client-name"),
    #             "local-classes": configs.get("local-classes"),
    #         }
    #         for k in keys:
    #             item[k] = metrics.get(k)
    #         round_metrics.append(item)

    #     return round_metrics

    def compile_clients_metrics(self, replies: Iterable[Message]) -> list:
        """
        Extracts history lists from ConfigRecord and scalar metrics from MetricRecord.
        """
        round_metrics = []

        for msg in replies:
            if not msg.has_content():
                continue

            # 1. Get the History Data (Lists) from ConfigRecord
            configs = msg.content.get("configs", ConfigRecord())

            # 2. Get the Scalar Data (Final epoch values) from MetricRecord
            metrics = msg.content.get("metrics", MetricRecord())

            # 3. Merge them into one dictionary for your parser
            # Note: We convert Record types to standard dicts for easier handling
            item = dict(configs)  # Start with history data

            # Add scalars with a prefix or distinct name to avoid key collisions
            # e.g. "final-train-acc"
            for k, v in metrics.items():
                # item[f"final-{k}"] = v
                item[f"{k}"] = v

            round_metrics.append(item)

        return round_metrics


def construct_messages_per_node(
    content_and_id: list[tuple[int, dict]],
    record: type[ConfigRecord] | type[ArrayRecord] | type[MetricRecord] = ConfigRecord,
) -> Iterable[Message]:
    messages = []
    # print("\nConstruct msg:")
    for item in content_and_id:
        node_id = item[0]  # this should be int
        content = item[1]  # this should be dict
        # print(f"node id: {node_id}\ncontent: {content}\n")
        msg = Message(
            content=RecordDict({"config": record(content)}),
            dst_node_id=node_id,
            message_type=MessageType.TRAIN,
        )
        messages.append(msg)
    return messages


def send_to_node(grid: Grid, messages: Iterable[Message]):
    replies = grid.send_and_receive(messages)
    return replies


###############################################################################


class CustomScheduler:
    def __init__(
        self,
        lr: float,
        patience: int,
        mode: Literal["min", "max"] = "min",
        threshold: float = 1e-4,
        factor: float = 0.1,
    ):
        self.lr = lr
        self.patience = patience
        self.mode = mode
        self.factor = factor
        self.threshold = threshold
        self.best = None
        self.counter = 0

    def step(self, loss: float):
        if self.best is None:
            self.best = loss
            return self.lr
        else:
            if self.is_improved(loss):
                self.counter = 0
                self.best = loss
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.lr *= self.factor
            self.counter = 0
            return self.lr

    def is_improved(self, current: float):
        delta = self.best - current
        print(f"\n scheduler delta: {delta}\n")
        if self.mode == "min":
            return delta > self.threshold
        else:
            return (-delta) > self.threshold


###############################################################################


class GlobalEvaluation:
    def __init__(self, model, dev, test_dataloader):
        self.model = model
        self.dev = dev
        self.test_dataloader = test_dataloader

    def __call__(self, *args, **kwds):
        from model_functions import test

        for arg in args:
            if isinstance(arg, int):
                server_round = arg
            elif isinstance(arg, ArrayRecord):
                arrays = arg

        device = torch.device(self.dev)

        self.model.load_state_dict(arrays.to_torch_state_dict())
        self.model.to(device)

        # Evaluate the global model on the test set
        test_acc, test_loss = test(
            self.model, self.test_dataloader, torch.nn.CrossEntropyLoss()
        )

        # Return the evaluation metrics
        return MetricRecord(
            {"accuracy": test_acc, "loss": test_loss, "round": server_round}
        )


###############################################################################


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Class for a Convolutional Neural Network
    """

    def __init__(
        self,
        out_features: int,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(224, 6, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.mx_pool = nn.MaxPool2d(2, 2)
        # self.adp_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(filters_nbr * 1 * 1, 120)
        out_1 = int(out_len(224, fltr_len=kernel_size) / 2)
        out_2 = int(out_len(out_1, fltr_len=kernel_size) / 2)
        self.fc1 = nn.Linear(16 * out_2 * out_2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mx_pool(x)
        x = F.relu(self.conv2(x))
        # x = self.adp_pool(x)
        x = self.mx_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def out_len(input_len: int, fltr_len: int, padding: int = 0, stride: int = 1):
    return 1 + ((input_len - fltr_len + 2 * padding) / stride)
