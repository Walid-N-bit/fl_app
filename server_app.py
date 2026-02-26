import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM
from CustomClasses import ConvolutionalNeuralNetwork as CNN, CustomStrat
from flwr.app import ConfigRecord
from utils import load_centralized_dataset, test, parse_raw_metrics, metrics_to_csv

from datetime import datetime
from ast import literal_eval

# from model_params import *

server = ServerApp()


def global_evaluate(model: CNN, server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Load the model and initialize it with the received weights

    # model = CNN(
    #     in_channels=IMG_C,
    #     out_channels=OUTPUT_CHANNELS,
    #     kernel_size=KERNEL_SIZE,
    #     out_features=len(CLASSES),
    # ).to(DEVICE)

    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(dataset=DATASET_ID)

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


@server.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    IMG_C = context.run_config["img_c"]
    IMG_H = context.run_config["img_h"]
    OUTPUT_CHANNELS = literal_eval(context.run_config["out_channels"])
    KERNEL_SIZE = context.run_config["kernel_size"]
    CLASSES = literal_eval(context.run_config["classes"])
    global DATASET_ID
    DATASET_ID = context.run_config["dataset_id"]
    ABS_PATH = context.node_config["abs_path"]

    start_time = datetime.now()

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    local_epochs = context.run_config["local-epochs"]
    momentum = context.run_config["momentum"]

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load global model

    global_model = CNN(
        in_channels=IMG_C,
        out_channels=OUTPUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        out_features=len(CLASSES),
        img_h=IMG_H,
    ).to(DEVICE)

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
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        # evaluate_fn=global_evaluate,
    )

    final_metrics = global_evaluate(global_model, num_rounds, result.arrays)
    print(f"Final accuracy: {final_metrics['accuracy']}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()

    time = datetime.now().strftime("%H:%M--%d-%m-%Y")
    model_path = f"{ABS_PATH}models/global_model_{time}.pt"

    torch.save(state_dict, model_path)

    print("\nSaving Clients Metrics Data...")
    metrics = parse_raw_metrics(evaluate_replies)
    client_data_path = (
        f"clients_data/lr:{lr}-epochs:{local_epochs}-momentum:{momentum}/{time}.csv"
    )
    metrics_to_csv(metrics, path=client_data_path)

    elapsed_time = datetime.now() - start_time
    print(f"\n##################################")
    print(f"\nTotal Elapsed Time: {elapsed_time}")
    print(f"\n##################################")
