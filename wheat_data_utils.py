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
    data = pd.read_csv(csv_path)
    classes_names = sorted(set(data["class_name"]))
    labels_map = {name: i for name, i in enumerate(classes_names, 0)}
    return labels_map


def img_labels(data_file: str):
    img_labels = []
    labels_map = labels_map_from_csv(data_file)
    data = pd.read_csv(data_file)
    for row in data.itertuples():
        img_labels.append((row.name, get_label(labels_map, row.class_name)))
    return pd.DataFrame(img_labels)


def imgs_data_to_csv(
    dataset_path: str,
    train_folders: list[str],
    test_folders: list[str],
):

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
            class_name = match.group(1)
            data.append(dict(name=name, class_name=class_name, path=path.as_posix()))
        return data

    def save_data(data: list, name: str):
        data_pd = pd.DataFrame(data)
        data_pd.to_csv(f"{dataset_path}/{name}.csv")

    all_train_dirs = get_dirs(train_folders)
    all_test_dirs = get_dirs(test_folders)
    clean_train_dirs = clean_dirs(all_train_dirs)
    clean_test_dirs = clean_dirs(all_test_dirs)
    train_data = data_prep(clean_train_dirs)
    test_data = data_prep(clean_test_dirs)
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


def get_class_weights(data_path: str, subset_indices: list = []) -> torch.Tensor:
    data = pd.read_csv(data_path)
    if len(subset_indices) > 0:
        data = data.iloc[subset_indices]
    _, counts = np.unique(data["class_name"], return_counts=True)
    class_counts = torch.tensor(counts)
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts * len(class_counts))
    # return class_weights / class_weights.sum()
    return class_weights


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


class WheatImgDataset(Dataset):

    def __init__(self, data_file, transform=None, target_transform=None):
        self.img_labels = img_labels(data_file)
        self.data_dir = pd.read_csv(data_file).to_numpy()
        self.transform = transform
        self.target_transform = target_transform
        self.classes = labels_map_from_csv(data_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.data_dir[idx, 3]

        # using PIL because torchvision.transforms expect it
        print("\n get image: ", img_path)
        image = Image.open(img_path).convert("RGB")

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
