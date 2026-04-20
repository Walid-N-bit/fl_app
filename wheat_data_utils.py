from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch
from torch.utils.data import WeightedRandomSampler
from utils import generate_labels_map


def get_label(label_map: dict, class_name: str) -> int:
    """
    get a label of a given class name

    :param label_map: label map
    :type label_map: dict
    :param class_name: target class name
    :type class_name: str
    :return: label
    :rtype: int
    """
    label = [
        key for key, value in label_map.items() if value.lower() == class_name.lower()
    ]
    return label[0]


def labels_map_from_csv(csv_path: str) -> dict:
    """
    retrieve labels map from a csv file

    :param csv_path: Description
    :type csv_path: str
    :return: Description
    :rtype: dict
    """
    file_exists = os.path.exists(csv_path)
    if file_exists:
        data = pd.read_csv(csv_path)
        classes_names = sorted(set(data["class_name"]))
        labels_map = {name: i for name, i in enumerate(classes_names, 0)}
        return labels_map
    else:
        return {}


def img_labels(data_file: str, labels_map: dict | None = None):
    """
    get a dataframe of each image and its corresponding label

    :param data_file: path of the .csv data file
    :type data_file: str
    :param labels_map: dataset's label's map
    :type labels_map: dict
    :return: dataframe of each image and its label
    :rtype: DataFrame
    """
    img_labels = []
    file_exists = os.path.exists(data_file)
    if file_exists:
        l_map = labels_map if labels_map else labels_map_from_csv(data_file)
        data = pd.read_csv(data_file)
        for row in data.itertuples():
            img_labels.append((row.name, get_label(l_map, row.class_name)))
        return pd.DataFrame(img_labels)
    else:
        return {}


def imgs_data_to_csv(
    dataset_path: str,
    train_folders: list[str],
    test_folders: list[str],
):
    """
    scan data folders and generate .csv files for training and testing data.

    :param dataset_path: folder containing all data folders
    :type dataset_path: str
    :param train_folders: paths of training data
    :type train_folders: list[str]
    :param test_folders: paths of testing (evaluation) folders
    :type test_folders: list[str]
    """

    def get_dirs(folders):
        dirs = []
        for f in folders:
            paths = glob.glob(f"{Path(dataset_path) / f / '**'}", recursive=True)
            dirs.extend(paths)
        return dirs

    def clean_dirs(dirs: list[str]):
        new_dirs = []
        for d in dirs:
            is_included = ".png" in d
            if is_included:
                new_dirs.append(d)
        return new_dirs

    def data_prep(paths: list[str]):
        data = []
        for path in paths:
            match = re.search(r"/([^/]+)/segmented_256_lcr_png", path)
            path = Path(path)
            name = path.name
            class_name = match.group(1).lower()
            data.append(dict(name=name, class_name=class_name, path=path.as_posix()))
        return pd.DataFrame(data)

    def add_labels(df: pd.DataFrame):
        classes = sorted(df["class_name"].unique())
        l_m = generate_labels_map(classes)
        labels = []
        for v in df["class_name"]:
            labels.append(get_label(l_m, v))
        df["label"] = labels
        return df

    def save_data(dataframe: pd.DataFrame, name: str):
        path = f"{dataset_path}/{name}.csv"
        dataframe.to_csv(path, index=False)

    all_train_dirs = get_dirs(train_folders)
    all_test_dirs = get_dirs(test_folders)
    clean_train_dirs = clean_dirs(all_train_dirs)
    clean_test_dirs = clean_dirs(all_test_dirs)
    train_data = data_prep(clean_train_dirs)
    test_data = data_prep(clean_test_dirs)
    # train_data = add_labels(train_data)
    # test_data = add_labels(test_data)
    save_data(train_data, "train")
    save_data(test_data, "test")


def data_summary(data_path: str) -> dict:
    data = pd.read_csv(data_path)
    df = pd.DataFrame(data)
    class_names = sorted(set(df["class_name"].tolist()))
    total_size = len(df.index)
    size_per_class = {}
    for c in class_names:
        count = sum(df["class_name"] == c)
        size_per_class.update({c: count})
    stats = dict(total_size=total_size, size_per_class=size_per_class)
    return stats


def compute_class_weights(data_summary: list[int]) -> torch.Tensor:
    class_counts = torch.tensor(data_summary)
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts * len(class_counts))
    # return class_weights / class_weights.sum()
    return class_weights


def get_class_weights(data_path: str, subset_indices: list = []) -> torch.Tensor:
    data = pd.read_csv(data_path)
    if len(subset_indices) > 0:
        data = data.iloc[subset_indices]
    _, counts = np.unique(data["class_name"], return_counts=True)
    weights = compute_class_weights(counts)
    return weights


def oversampler(data_path: str, subset_indices: list = []) -> WeightedRandomSampler:

    from collections import Counter

    data = pd.read_csv(data_path)
    if len(subset_indices) > 0:
        data = data.iloc[subset_indices]
    class_counts = Counter(data["class_name"])  # counter dict
    class_sample_weights = {c: 1.0 / count for c, count in class_counts.items()}
    # sample_weights = [0] * len(data)
    sample_weights = []

    for _, item in data.iterrows():
        class_name = item["class_name"]
        class_weight = class_sample_weights.get(class_name)
        sample_weights.append(class_weight)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def save_csv(
    path: str,
    data: pd.DataFrame,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path)


def update_csv(file_path: str, new_row: pd.DataFrame):
    old_rows = pd.read_csv(file_path, index_col=0)
    df = pd.concat([old_rows, new_row], ignore_index=True)
    df.to_csv(file_path)


def data_summary(data_path: str) -> dict:
    data = pd.read_csv(data_path)
    df = pd.DataFrame(data)
    class_names = sorted(set(df["class_name"].tolist()))
    total_size = len(df.index)
    size_per_class = {}
    for c in class_names:
        count = sum(df["class_name"] == c)
        size_per_class.update({c: count})
    stats = dict(total_size=total_size, size_per_class=size_per_class)
    return stats


# ==============================================
# primary class for the wheat data
# ==============================================
class WheatImgDataset(Dataset):

    def __init__(
        self,
        data_file,
        transform=None,
        target_transform=None,
        labels_map: dict | None = None,
    ):
        self.data_file = data_file
        self.img_labels = img_labels(data_file, labels_map)
        self.data_dir = pd.read_csv(data_file, index_col=0)
        # self.data_dir = pd.read_csv(data_file, index_col=0).to_numpy()
        self.transform = transform
        self.target_transform = target_transform
        self.classes = labels_map if labels_map else labels_map_from_csv(data_file)

    def change_class_labels(self, labels_map: dict):
        self.classes = labels_map
        self.img_labels = img_labels(self.data_file, labels_map)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = self.data_dir[idx, 3]
        img_path = self.data_dir.iloc[idx]["path"]

        # using PIL because torchvision.transforms expect it
        image = Image.open(img_path).convert("RGB")

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
