import os
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch.nn.functional as F
from flwr.serverapp.strategy import FedAvg, FedAvgM
from collections.abc import Iterable
from flwr.app import Message, MetricRecord, RecordDict

# from model_params import FLTRS_NBR, IMG_H, IMG_W
import time, io
from collections.abc import Callable
from logging import INFO

from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, log, MessageType
from flwr.server import Grid

from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info


class ImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomStrat(FedAvg):
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
    ) -> tuple[list[Iterable[Message]], list[Iterable[Message]], Result]:
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

        # added this to capture client metrics #
        # train_replies_all = []
        # eval_replies_all = []
        clients_train_metrics = {}
        clients_eval_metrics = {}
        ########################################

        # for current_round in range(0, num_rounds + 1):
        for current_round in range(1, num_rounds + 1):

            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

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

            # saving client-side train metrics
            # print(f"{len(train_replies) = }")
            # print(f"{type(train_replies) = }")
            # for reply in train_replies:
            #     print(f"{type(reply) = }")
            clients_train_metrics[current_round] = self.compile_clients_metrics(
                train_replies
            )
            # same way you can add client-side eval metrics

            ###############################

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

    def compile_clients_metrics(self, replies: Iterable[Message]) -> list:
        """
        parse replies data and return a dict for metrics and the client that produced them

        :param replies: message replies from clients
        :type replies: Iterable[Message]
        :return: data to be returned to server
        :rtype: dict
        """
        round_metrics = []
        for msg in replies:
            configs = msg.content["configs"]
            metrics = msg.content["metrics"]
            keys = metrics.keys()
            item = {
                "client-name": configs.get("client-name"),
                "local-classes": configs.get("local-classes"),
            }
            for k in keys:
                item[k] = metrics.get(k)
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


class GlobalEvaluation:
    def __init__(self, model, dev):
        self.model = model
        self.dev = dev

    def __call__(self, *args, **kwds):
        from model_functions import test
        from wheat_data_utils import WheatImgDataset
        from wheat_data_prep import TEST_DATA_PATH, data_loader
        from torchvision import transforms

        for arg in args:
            if isinstance(arg, int):
                server_round = arg
            elif isinstance(arg, ArrayRecord):
                arrays = arg

        device = torch.device(self.dev)
        pt_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.model.load_state_dict(arrays.to_torch_state_dict())
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Load entire test set (for CIFAR10)
        # test_dataloader = load_centralized_dataset(dataset=DATASET_ID)
        test_data = WheatImgDataset(TEST_DATA_PATH, pt_transforms)
        test_dataloader = data_loader(test_data, self.dev, 128)

        # Evaluate the global model on the test set
        test_acc, test_loss = test(
            self.model, test_dataloader, torch.nn.CrossEntropyLoss()
        )

        # Return the evaluation metrics
        return MetricRecord(
            {"accuracy": test_acc, "loss": test_loss, "server-round": server_round}
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
