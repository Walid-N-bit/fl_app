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


def cmd(input: str | list, passwd: bool = False, shell: bool = False) -> str:
    """
    take input and run as a command on shell. return output.

    :param input: input command
    :type input: str | list
    :param passwd: does the command require password input
    :type passwd: bool
    :param shell: set as true if you need shell functionalities (like > or --)
    :type shell: bool
    :return: command output
    :rtype: str
    """
    import subprocess
    from getpass import getpass

    if type(input) == str and not shell:
        input = input.split(" ")
    proc = subprocess.Popen(
        args=input,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
        text=True,
    )

    out = ""
    if passwd:
        pw = getpass()
        out, _ = proc.communicate(input=pw)
    else:
        out, _ = proc.communicate()
    return out


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
    if message.has_content():
        node_id = message.metadata.src_node_id
        metrics = message.content.metric_records["metrics"]
        metrics.update({"node_id": node_id})
        return metrics
    else:
        return {"metrics": None}


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


def readable_time(seconds: float):
    import time

    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def end_of_training_msg(time: float):
    msg = f"\nEnd of Training. Elapsed Time: {readable_time(time)}\n"
    bars = "".join(["#" for _ in range(len(msg))])
    print(bars)
    print(msg)
    print(bars)


def pick_mixer(name: str, classes: list):
    from torchvision.transforms import v2

    cutmix = v2.CutMix(num_classes=len(classes))
    mixup = v2.MixUp(num_classes=len(classes))
    cutmixup = v2.RandomChoice([cutmix, mixup])
    match name:
        case "cutmix":
            return cutmix
        case "mixup":
            return mixup
        case "cutmixup":
            return cutmixup
        case _:
            return None
