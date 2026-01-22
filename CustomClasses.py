import os
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch.nn.functional as F
from flwr.serverapp.strategy import FedAvg
from collections.abc import Iterable
from flwr.app import Message, MetricRecord


class ImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomStrat(FedAvg):
    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        metrics = super().aggregate_evaluate(self, server_round, replies)

        print("--------------------------------")
        print(metrics)
        print("--------------------------------")
        return metrics


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Class for a Convolutional Neural Network
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, out_features: int
    ):
        super().__init__()
        filters_nbr = 32
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(out_channels, filters_nbr, kernel_size)
        self.mx_pool = nn.MaxPool2d(2, 2)
        # self.adp_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(filters_nbr * 1 * 1, 120)
        out_1 = int(out_len(input_len=32, fltr_len=5) / 2)
        out_2 = int(out_len(out_1, fltr_len=5) / 2)
        self.fc1 = nn.Linear(filters_nbr * out_2 * out_2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mx_pool(x)
        x = F.relu(self.conv2(x))
        # x = self.adp_pool(x)
        x = self.mx_pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def out_len(input_len: int, fltr_len: int, padding: int = 0, stride: int = 1):
    return 1 + ((input_len - fltr_len + 2 * padding) / stride)
