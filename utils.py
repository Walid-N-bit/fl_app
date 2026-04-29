import os, csv, json
import torch
import pandas as pd
from typing import Any, Dict, List
from torch.utils.data import Dataset

from flwr.common import Message

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# DATA_COLUMNS = [
#     "epoch",
#     "features-lr",
#     "classifier-lr",
#     "train-acc",
#     "val-acc",
#     "train-loss",
#     "val-loss",
#     "train-time",
# ]


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


# def client_metrics(epochs: int, client_metrics: dict) -> dict:

#     name = client_metrics.get("client-name")
#     client_name = [name for _ in range(epochs)]
#     data = {"client-name": client_name}
#     for key in DATA_COLUMNS:
#         data.update({key: client_metrics.get(key)})
#     return data


# def round_metrics(epochs: int, round: int, round_metrics: list[dict]) -> dict:

#     data = []
#     rounds = [round for _ in range(epochs)]
#     for client in round_metrics:
#         # print("\nclient to be processed: ", client)
#         client_data = client_metrics(epochs, client)
#         client_data.update({"round": rounds})
#         data.append(client_data)
#         # print("\npre-extended data: ", data)
#     data = extend_dict(data)
#     return data


# def parse_raw_metrics(raw_metrics: dict[list[dict[str, list]]]) -> pd.DataFrame:
#     """
#     transform metrics into a dataframe

#     :param raw_metrics: metrics from clients
#     :type raw_metrics: list[dict]
#     :return: Description
#     :rtype: dict
#     """

#     data = []
#     epochs = len(raw_metrics.get(1)[0].get("epoch"))
#     # print(f"{epochs = }")
#     for round in raw_metrics:
#         round_data = round_metrics(epochs, round, raw_metrics[round])
#         data.append(round_data)
#     # print(f"\n{data = }")

#     data = extend_dict(data)
#     return pd.DataFrame(data)


# def parse_server_eval_metrics(metrics: dict[int, dict]) -> pd.DataFrame:
#     """
#     transform evaluation metrics from the server to pandas dataframe

#     :param metrics: raw metrics data
#     :type metrics: dict[int, dict]
#     :return: metrics dataframe
#     :rtype: DataFrame
#     """
#     keys = metrics.get(0).keys()
#     data = {k: [] for k in keys}
#     for i in metrics:
#         round_metrics = metrics.get(i)
#         for k in keys:
#             data[k].append(round_metrics.get(k))
#     return pd.DataFrame(data)


def parse_raw_metrics(raw_metrics: dict[int, list[dict[str, Any]]]) -> pd.DataFrame:
    """
    Transforms nested raw metrics into a Pandas DataFrame dynamically.
    Handles both list-based metrics (epochs) and scalar values (metadata) automatically.

    Args:
        raw_metrics: Dictionary mapping round numbers to lists of client metric dicts.
                     Client dicts contain both scalar values (e.g., 'client-name')
                     and list values (e.g., 'train-loss' per epoch).

    Returns:
        A flat Pandas DataFrame where each row corresponds to one epoch of one client.
    """
    rows = []

    for round_id, client_list in raw_metrics.items():
        for client_data in client_list:
            if not client_data:
                continue

            # 1. Separate scalars (metadata) from lists (time-series metrics)
            # We assume metrics that vary per epoch are lists, and constants are scalars.
            scalars = {k: v for k, v in client_data.items() if not isinstance(v, list)}
            series_data = {k: v for k, v in client_data.items() if isinstance(v, list)}

            # 2. Determine the number of epochs (steps) from the first available list
            # If no lists exist (e.g., only scalars returned), we create 1 row.
            num_steps = 0
            for v in series_data.values():
                if len(v) > 0:
                    num_steps = len(v)
                    break

            if num_steps == 0 and scalars:
                # Handle case where client only returned scalars
                row = {"round": round_id, **scalars}
                rows.append(row)
                continue

            # 3. Flatten the lists into individual rows
            for i in range(num_steps):
                row = {"round": round_id}

                # Add metadata (repeated for each epoch row)
                row.update(scalars)

                # Add the specific value for this epoch index 'i'
                for key, values in series_data.items():
                    # Check bounds to prevent errors if lists are uneven
                    row[key] = values[i] if i < len(values) else None

                rows.append(row)

    # 4. Create DataFrame
    df = pd.DataFrame(rows)

    # Optional: Reorder columns to put important identifiers first
    # This is dynamic but puts 'round', 'client-name' at the front if they exist
    priority_cols = ["round", "client-name"]
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_priority]

    return df[existing_priority + other_cols]


def parse_server_eval_metrics(metrics: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms server evaluation metrics into a DataFrame.

    Args:
        metrics: Dictionary mapping round numbers to metric dictionaries.

    Returns:
        A Pandas DataFrame with one row per round.
    """
    # The simplest way to handle dynamic dict->DataFrame conversion
    # We add the round key to the dictionary for each row
    rows = [{"round": r, **data} for r, data in metrics.items()]

    return pd.DataFrame(rows)


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


def save_arbitrary_json(path: str, **kwargs):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(path, "w") as f:
        json.dump(kwargs, f, indent=4)


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


def split_df_by_type(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into two based on column content types.

    Returns:
        df_scalars: Columns containing single numbers/strings (good for plots/stats).
        df_lists:   Columns containing lists/objects (good for inspection/exploding).
    """
    scalar_cols = []
    list_cols = []

    # Iterate over the first row to determine types
    # We assume the first row is representative of the whole column
    if not df.empty:
        first_row = df.iloc[0]
        for col in df.columns:
            # Check if the value is a list
            if isinstance(first_row[col], list):
                list_cols.append(col)
            else:
                scalar_cols.append(col)

    return df[scalar_cols], df[list_cols]
