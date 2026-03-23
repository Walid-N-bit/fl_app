import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM
from CustomClasses import CustomStrat
from flwr.app import ConfigRecord
from utils import parse_raw_metrics, metrics_to_csv

from datetime import datetime

# from ast import literal_eval

from model_functions import choose_model
from cifar10_data_prep import CIFAR10_CLASSES as cifar_classes

server = ServerApp()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prep_phase(strategy: CustomStrat, grid: Grid, arrays: ArrayRecord) -> list:
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
    prep_conf = MetricRecord({"prep-phase": 1})
    prep_replies = strategy.prepare(grid, arrays, prep_config=prep_conf)
    global_classes = set()
    for item in prep_replies:

        print("\nsource ", item.metadata.src_node_id)
        print("\ndestination ", item.metadata.dst_node_id)
        print("\ncontent ", item.content)

        client_classes = item.content.get("metrics").get("local-classes")
        global_classes.update(set(client_classes))

    return sorted(list(global_classes))


def labels_per_client(global_classes:list, grid:Grid):
    labels_map = {i: c for i, c in enumerate(global_classes)}

    pass


# def global_evaluate(model: CNN, server_round: int, arrays: ArrayRecord) -> MetricRecord:
#     """Evaluate model on central data."""

#     model.load_state_dict(arrays.to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load entire test set
#     test_dataloader = load_centralized_dataset(dataset=DATASET_ID)

#     # Evaluate the global model on the test set
#     test_loss, test_acc = test(model, test_dataloader, device)

#     # Return the evaluation metrics
#     return MetricRecord({"accuracy": test_acc, "loss": test_loss})


@server.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

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

    configs = {
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
        "prep": False,
    }
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
    golobal_classes = prep_phase(strategy, grid, temp_arrays)
    out_features = len(golobal_classes)
    global_model = choose_model(model_name, freeze, out_features).to(DEVICE)
    arrays = ArrayRecord(global_model.state_dict())

    # Start strategy, run FedAvg for `num_rounds`
    train_replies, evaluate_replies, result = strategy.start(
        timeout=1e10,
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(configs),
        num_rounds=num_rounds,
        # evaluate_fn=global_evaluate,
    )

    # final_metrics = global_evaluate(global_model, num_rounds, result.arrays)
    # print(f"Final accuracy: {final_metrics['accuracy']}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()

    # time = datetime.now().strftime("%H:%M-%d/%m/%Y")
    # model_path = f"/home/wnouicer24/thesis/fl_app/models/global_model_{time}.pt"

    # torch.save(state_dict, model_path)

    print("\nSaving Clients Metrics Data...")
    t_metrics = parse_raw_metrics(train_replies)
    e_metrics = parse_raw_metrics(evaluate_replies)
    print("\nparsed train metrics:\n", t_metrics)
    print("\nparsed eval metrics:\n", e_metrics)
    # client_data_path = (
    #     f"clients_data/lr:{lr}-epochs:{local_epochs}-momentum:{momentum}/{time}.csv"
    # )
    # metrics_to_csv(metrics, path=client_data_path)
