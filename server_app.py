import torch
from flwr.app import ArrayRecord, Context, MetricRecord, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM
from CustomClasses import CustomStrat, GlobalEvaluation
from utils import (
    cmd,
    save_pkl,
    parse_raw_metrics,
    parse_server_eval_metrics,
    generate_labels_map,
)

from datetime import datetime
import os
from typing import Literal

from model_functions import choose_model, eval_per_class
from cifar10_data_prep import CIFAR10_CLASSES

server = ServerApp()

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEV)


def prep_phase(
    strategy: CustomStrat, grid: Grid, arrays: ArrayRecord
) -> tuple[list, list]:
    """
    send flag to clients to signify preparation phase of the training.
    receive and compile local classes from each client into a list of global classes.

    :param strategy: Description
    :type strategy: CustomStrat
    :param grid: Description
    :type grid: Grid
    :param arrays: Description
    :type arrays: ArrayRecord
    :return: Description
    :rtype: set
    """
    # prep_conf = MetricRecord({"prep-phase": 1})
    prep_conf = ConfigRecord({"prep-phase": 1})
    prep_replies = strategy.prepare(grid, arrays, prep_config=prep_conf)
    global_classes = set()
    clients_configs = []
    for item in prep_replies:

        # print("\nsource ", item.metadata.src_node_id)
        # print("\ndestination ", item.metadata.dst_node_id)
        # print("\ncontent ", item.content)
        client_conf = item.content.get("config")
        client_classes = client_conf.get("local-classes")
        clients_configs.append(client_conf)
        global_classes.update(set(client_classes))

    return sorted(list(global_classes)), clients_configs


def labels_map_per_client(global_classes: list, configs: list[dict]):

    labels_map = {i: c for i, c in enumerate(global_classes)}
    print("\nGlobal labels map: ", labels_map)
    contents = []
    for item in configs:
        node_name = item.get("node-name")
        node_id = item.get("node-id")
        node_classes = item.get("local-classes")
        node_labels_map = {k: v for k, v in labels_map.items() if (v in node_classes)}
        contents.append((int(node_id), {"labels": list(node_labels_map.keys())}))
    print("\nContents: ", contents)
    print(" ")
    return contents


def pick_test_dataloader(dataset_name: Literal["cifar10", "wheat"]):
    from wheat_data_prep import TESTING_DATA, data_loader as wheat_loader
    from cifar10_data_prep import CIFAR10_TEST, loader as cifar_loader

    if dataset_name == "wheat":
        test_dataloader = wheat_loader(TESTING_DATA, DEV, 128)
    elif dataset_name == "cifar10":
        test_dataloader = cifar_loader(CIFAR10_TEST, 128)

    return test_dataloader


@server.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    from CustomClasses import send_to_node, construct_messages_per_node

    model_name = context.run_config["model-name"]
    freeze = context.run_config["freeze"]
    batch_size = context.run_config["batch-size"]
    use_sampler = context.run_config["use-sampler"]
    num_workers = context.run_config["num-workers"]
    features_lr = context.run_config["features-lr"]
    classifier_lr = context.run_config["classifier-lr"]
    weight_decay = context.run_config["weight-decay"]
    sch_patience = context.run_config["sch-patience"]
    use_weights = context.run_config["use-weights"]
    epochs = context.run_config["local-epochs"]
    dataset_name = context.run_config["dataset-name"]
    mixer = context.run_config["mixer"]
    proximal_mu = context.run_config["proximal-mu"]

    start_time = datetime.now()

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    momentum = context.run_config["momentum"]

    # Load global model
    # if dataset_name == "wheat":
    #     # from wheat_data_prep import CLASSES as wheat_classes

    #     # out_features = len(wheat_classes)
    #     out_features = 1
    # elif dataset_name == "cifar10":
    #     out_features = len(cifar_classes)

    temp_model = choose_model("cnn", freeze, 1).to(DEVICE)

    ## model_exists = file_exists(output_path)
    # if model_exists:
    #     global_model.load_state_dict(torch.load(output_path, weights_only=True))

    temp_arrays = ArrayRecord(temp_model.state_dict())

    ## Initialize FedAvg strategy
    # strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    # strategy = CustomStrat(
    #     fraction_evaluate=fraction_evaluate, server_momentum=momentum
    # )

    # for the custom strat based on FedAvg
    # strategy = CustomStrat(fraction_evaluate=fraction_evaluate)

    # for the custom strat based on FedProx
    strategy = CustomStrat(fraction_evaluate=fraction_evaluate, proximal_mu=proximal_mu)

    # prepare for training by receiving client arrays
    global_classes, all_metrics = prep_phase(strategy, grid, temp_arrays)
    labels_maps = labels_map_per_client(global_classes, all_metrics)
    messages_to_clients = construct_messages_per_node(labels_maps)
    labels_msg_replies = send_to_node(grid, messages_to_clients)

    # print replies for sent labels
    for item in labels_msg_replies:
        print(
            f"\n--> {item.content.get("config").get("node-name")} have received assigned labels successfully."
        )
    print("")

    out_features = len(global_classes)
    global_model = choose_model(model_name, freeze, out_features).to(DEVICE)
    arrays = ArrayRecord(global_model.state_dict())

    print("\n### global classes: ", global_classes)

    # compile training configs
    train_configs = {
        "model-name": model_name,
        "freeze": freeze,
        "batch-size": batch_size,
        "use-sampler": use_sampler,
        "num-workers": num_workers,
        "features-lr": features_lr,
        "classifier-lr": classifier_lr,
        "weight-decay": weight_decay,
        "sch-patience": sch_patience,
        "use-weights": use_weights,
        "local-epochs": epochs,
        "dataset-name": dataset_name,
        "mixer": mixer,
        "out-features": out_features,
        # "proximal-mu": proximal_mu,
    }
    # Start strategy, run FedAvg for `num_rounds`
    test_dataloader = pick_test_dataloader(dataset_name)
    train_replies, _, result = strategy.start(
        timeout=1e10,
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_configs),
        num_rounds=num_rounds,
        evaluate_fn=GlobalEvaluation(global_model, DEV, test_dataloader),
    )

    # final_metrics = global_evaluate(global_model, num_rounds, result.arrays)
    # aggregated_metrics = result.evaluate_metrics_serverapp
    # print(f"\nAggregated metrics:\n{aggregated_metrics}\n")

    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()
    global_model.load_state_dict(state_dict)
    g_labels_map = generate_labels_map(global_classes)
    eval_per_class(test_dataloader, global_model, out_features, g_labels_map)

    time = datetime.now().strftime("%H:%M-%d.%m.%Y")

    model_path = f"/root/data/models/{dataset_name}_{model_name}_epochs:{epochs}_f-lr:{features_lr}_c-lr:{classifier_lr}_batch-size:{batch_size}_aug:{mixer}_{time}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("\nSaving final model to disk...")
    torch.save(state_dict, model_path)

    print("\n\nSaving Clients Metrics Data...\n")
    data_name = f"{model_name}_epochs:{epochs}_f-lr:{features_lr}_c-lr:{classifier_lr}_batch-size:{batch_size}_aug:{mixer}_{time}"
    raw_data_path = f"/root/data/metrics/{dataset_name}/{data_name}.pkl"
    csv_data_path = f"/root/data/metrics/{dataset_name}/{data_name}.csv"
    save_pkl(raw_data_path, train_replies)
    data_df = parse_raw_metrics(train_replies)
    data_df.to_csv(csv_data_path, index=False)

    print("\n\nSaving Server Evaluation Metrics Data...\n")
    raw_eval_data_path = f"/root/data/metrics/{dataset_name}/{data_name}_eval.pkl"
    csv_eval_data_path = f"/root/data/metrics/{dataset_name}/{data_name}_eval.csv"
    print(result.evaluate_metrics_serverapp)
    save_pkl(raw_eval_data_path, result.evaluate_metrics_serverapp)
    eval_data_df = parse_server_eval_metrics(result.evaluate_metrics_serverapp)
    eval_data_df.to_csv(csv_eval_data_path, index=False)
