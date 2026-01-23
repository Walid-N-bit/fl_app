import os
import csv

import torch
import pandas as pd
from torch.utils.data import Dataset

from flwr.common import Message

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset: str):

    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import IidPartitioner

    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def train(model, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    model.to(device)  # move model to GPU if available
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = loss_func(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(model, testloader, device):
    """Validate the model on the test set."""
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss += loss_func(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def load_centralized_dataset(dataset: str):
    """Load test set and return dataloader."""
    from datasets import load_dataset

    # Load entire test set
    test_dataset = load_dataset(dataset, split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def file_exists(path: str):
    """
    create path directory if it doesn't exist.
    check if path file exists or not. return true if exists, false otherwise.

    :param path: path to file
    :type path: str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    return file_exists


def image_shape(image: torch.Tensor):

    channels, height, width = tuple(image.shape)
    return channels, height, width


def node_metrics(message: Message) -> dict:
    node_id = message.metadata.src_node_id
    metrics = message.content.metric_records["metrics"]
    metrics.update({"node_id": node_id})
    return metrics


def round_metrics(replies: list[Message]) -> list[dict]:
    data = []
    for reply in replies:
        reply = node_metrics(reply)
        data.append(reply)
    return data


def parse_raw_metrics(raw_metrics: list[list[Message]]) -> list[dict]:
    data: list[list[dict]] = []
    final_data = []
    for item in raw_metrics:
        dict_item = round_metrics(item)
        data.append(dict_item)
    for i, round in enumerate(data, 1):
        for item in round:
            item.update({"round": i})
            final_data.append(item)

    return final_data


def metrics_to_csv(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    if len(data) > 0:
        fields = list(data[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(data)


def save_csv(fields: list, data: list, path: str):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(data)


def save_txt(data, path="logs.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)


from CustomClasses import ConvolutionalNeuralNetwork as CNN
from model_params import *


def test_img(image):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN(
        in_channels=IMG_C,
        out_channels=OUTPUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        out_features=len(CLASSES),
    ).to(DEVICE)

    model.eval()

    pass
