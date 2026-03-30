import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM
from CustomClasses import CustomStrat, GlobalEvaluation
from flwr.app import ConfigRecord
from utils import parse_raw_metrics, metrics_to_csv

from datetime import datetime

from model_functions import choose_model
from cifar10_data_prep import CIFAR10_CLASSES as cifar_classes

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


# def global_evaluate(server_round: int, arrays: ArrayRecord, model=None) -> MetricRecord:
#     """Evaluate model on central data."""
#     from model_functions import test
#     from wheat_data_utils import WheatImgDataset
#     from wheat_data_prep import TEST_DATA_PATH, data_loader
#     from torchvision import transforms

#     pt_transforms = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     model.load_state_dict(arrays.to_torch_state_dict())
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(DEVICE)

#     # Load entire test set (for CIFAR10)
#     # test_dataloader = load_centralized_dataset(dataset=DATASET_ID)
#     test_data = WheatImgDataset(TEST_DATA_PATH, pt_transforms)
#     test_dataloader = data_loader(test_data, DEV, 128)

#     # Evaluate the global model on the test set
#     test_loss, test_acc = test(model, test_dataloader, torch.nn.CrossEntropyLoss())

#     # Return the evaluation metrics
#     return MetricRecord(
#         {"accuracy": test_acc, "loss": test_loss, "server-round": server_round}
#     )


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
    strategy = CustomStrat(fraction_evaluate=fraction_evaluate)

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
    }
    # Start strategy, run FedAvg for `num_rounds`
    train_replies, evaluate_replies, result = strategy.start(
        timeout=1e10,
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_configs),
        num_rounds=num_rounds,
        evaluate_fn=GlobalEvaluation(global_model, DEV),
    )

    # final_metrics = global_evaluate(global_model, num_rounds, result.arrays)
    aggregated_metrics = result.evaluate_metrics_serverapp
    print(f"Aggregated metrics: {aggregated_metrics}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()

    time = datetime.now().strftime("%H:%M-%d/%m/%Y")
    model_path = f"/root/data/compressed_images_wheat/models/{dataset_name}_{model_name}_epochs:{epochs}_batch-size:{batch_size}_aug:{mixer}_{time}.pt"

    torch.save(state_dict, model_path)

    print("\nSaving Clients Metrics Data...")
    t_metrics = parse_raw_metrics(train_replies)
    e_metrics = parse_raw_metrics(evaluate_replies)
    print("\nparsed train metrics:\n", t_metrics)
    print("\nparsed eval metrics:\n", e_metrics)

    metrics_data_path = f"/root/data/metrics/{dataset_name}_{model_name}_epochs:{epochs}_batch-size:{batch_size}_aug:{mixer}_{time}.pt"

    # metrics_to_csv(metrics, path=metrics_data_path)
