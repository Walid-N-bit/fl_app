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

DATA_COLUMNS = [
    "epoch",
    "features-lr",
    "classifier-lr",
    "train-acc",
    "val-acc",
    "train-loss",
    "val-loss",
    "train-time",
]


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


def extend_dict(dicts: list[dict[str, list]]) -> dict:
    from copy import deepcopy

    data = deepcopy(dicts)[0]
    other_dicts = deepcopy(dicts)[1:]
    for item in other_dicts:
        # print(f"\n{item = }")
        for key in item.keys():
            # print(f"\n{data[key] = }")
            data[key].extend(item.get(key))
    return data


def client_metrics(epochs: int, client_metrics: dict) -> dict:

    name = client_metrics.get("client-name")
    client_name = [name for _ in range(epochs)]
    data = {"client-name": client_name}
    for key in DATA_COLUMNS:
        data.update({key: client_metrics.get(key)})
    return data


def round_metrics(epochs: int, round: int, round_metrics: list[dict]) -> dict:

    data = []
    rounds = [round for _ in range(epochs)]
    for client in round_metrics:
        # print("\nclient to be processed: ", client)
        client_data = client_metrics(epochs, client)
        client_data.update({"round": rounds})
        data.append(client_data)
        # print("\npre-extended data: ", data)
    data = extend_dict(data)
    return data


def parse_raw_metrics(raw_metrics: dict[list[dict[str, list]]]) -> pd.DataFrame:
    """
    transform metrics into a dataframe

    :param raw_metrics: metrics from clients
    :type raw_metrics: list[dict]
    :return: Description
    :rtype: dict
    """

    data = []
    epochs = len(raw_metrics.get(1)[0].get("epoch"))
    # print(f"{epochs = }")
    for round in raw_metrics:
        round_data = round_metrics(epochs, round, raw_metrics[round])
        data.append(round_data)
    # print(f"\n{data = }")

    data = extend_dict(data)
    return pd.DataFrame(data)


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


def save_pkl(path: str, data):
    """
    save raw data into .pkl files.

    :param path: file path
    :type path: str
    :param data: data to be saved
    """
    import pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path: str):
    """
    load data from a .pkl file.

    :param path: file path
    :type path: str
    :return: raw data
    :rtype: Any
    """
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def readable_time(seconds: float):
    import time

    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def end_of_training_msg(time: float):
    msg = f"\nEnd of Training. Elapsed Time: {readable_time(time)}\n"
    bars = "".join(["#" for _ in range(len(msg))])
    print(bars)
    print(msg)
    print(bars)


def pick_mixer(name: str, num_classes: int):
    from torchvision.transforms import v2

    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
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
