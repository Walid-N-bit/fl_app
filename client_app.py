from utils import image_shape, load_data
from utils import train as train_fn
from utils import test as test_fn
from CustomClasses import ConvolutionalNeuralNetwork, ImageDataset
import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

client = ClientApp()

CNN = ConvolutionalNeuralNetwork


# # -----------------------------------------------------------#
# # -------------------for custom datasets---------------------#
# TRAIN_PATH = "data/train/"
# TEST_PATH = "data/test/"

# TRAIN_DATA = ImageDataset(annotations_dir="data/Training_set.csv", img_dir=TRAIN_PATH)
# TEST_DATA = ImageDataset(annotations_dir="data/Testing_set.csv", img_dir=TEST_PATH)


# IMAGE, _ = TRAIN_DATA[0]


# # -----------------------------------------------------------#


DATASET_ID = "uoft-cs/cifar10"
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MODEL_PATH = "models/local_model.pth"

EPOCHS = 20


@client.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = CNN(
        in_channels=1, out_channels=3, kernel_size=5, out_features=len(CLASSES)
    ).to(DEVICE)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, DATASET_ID)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        DEVICE,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@client.test()
def test():
    pass


@client.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = CNN(
        in_channels=1, out_channels=3, kernel_size=5, out_features=len(CLASSES)
    ).to(DEVICE)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, DATASET_ID)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        DEVICE,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
