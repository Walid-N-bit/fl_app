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
from wheat_data_prep import CLASSES

server = ServerApp()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    configs = {
        "model-name": model_name,
        "freeze": freeze,
        "batch-size": batch_size,
        "use-sampler": use_sampler,
        "num-workers": 0,
        "features-lr": features_lr,
        "classifier-lr": classifier_lr,
        "weight-decay": weight_decay,
        "sch-patience": sch_patience,
        "use-weights": use_weights,
        "local-epochs": epochs,
    }
    start_time = datetime.now()

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    momentum = context.run_config["momentum"]

    # Load global model

    global_model = choose_model(model_name, freeze, len(CLASSES)).to(DEVICE)

    ## model_exists = file_exists(output_path)
    # if model_exists:
    #     global_model.load_state_dict(torch.load(output_path, weights_only=True))

    arrays = ArrayRecord(global_model.state_dict())

    ## Initialize FedAvg strategy
    # strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    # strategy = CustomStrat(
    #     fraction_evaluate=fraction_evaluate, server_momentum=momentum
    # )
    strategy = CustomStrat(fraction_evaluate=fraction_evaluate)
    # Start strategy, run FedAvg for `num_rounds`
    evaluate_replies, result = strategy.start(
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
    metrics = parse_raw_metrics(evaluate_replies)
    print("\nraw metrics:\n", evaluate_replies)
    print("\nparsed metrics:\n", metrics)
    # client_data_path = (
    #     f"clients_data/lr:{lr}-epochs:{local_epochs}-momentum:{momentum}/{time}.csv"
    # )
    # metrics_to_csv(metrics, path=client_data_path)
