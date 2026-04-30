import os, csv, json, time
import torch
import pandas as pd
from typing import Any, Dict, List
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


def bordered_print(text: str):
    print(f"\n{'='*len(text)}=")
    print(f" {text}")
    print(f"{'='*len(text)}=\n")


def readable_time(seconds: float):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def end_of_training_msg(time: float):
    msg = f"\nEnd of Training. Elapsed Time: {readable_time(time)}\n"
    bars = f"{'=' * len(msg)}"
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


def generate_labels_map(class_names: list[str]) -> dict[int, str]:
    """
    create a labels map for a list of classes names.

    :param class_names: classes
    :type class_names: list[str]
    :return: labels map
    :rtype: dict[int, str]
    """
    class_names = sorted(class_names)
    lm = {i: c.lower() for i, c in enumerate(class_names)}
    return lm


def get_model_size(model):
    """
    calculate the size of a model in MB

    :param model: model
    :return: size in MB
    :rtype: float
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
