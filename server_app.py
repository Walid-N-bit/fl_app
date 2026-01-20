from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
import torch
from flwr.serverapp.strategy import FedAvg
from CustomClasses import ConvolutionalNeuralNetwork as CNN
from flwr.app import ArrayRecord, ConfigRecord
from utils import load_centralized_dataset, test

server = ServerApp()

DATASET_ID = "uoft-cs/cifar10"  # shape: 3, 32, 32
C, H, W = 3, 32, 32

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model and initialize it with the received weights
    model = CNN(
        in_channels=C, out_channels=3, kernel_size=5, out_features=len(CLASSES)
    ).to(DEVICE)

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

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load global model
    global_model = CNN(
        in_channels=C, out_channels=3, kernel_size=5, out_features=len(CLASSES)
    ).to(DEVICE)

    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        # evaluate_fn=global_evaluate,
    )

    final_metrics = global_evaluate(num_rounds, result.arrays)
    print(f"Final accuracy: {final_metrics['accuracy']}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
