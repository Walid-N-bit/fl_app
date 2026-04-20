from wheat_data_utils import WheatImgDataset, oversampler, data_summary
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
from typing import Literal
from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path
from utils import cmd
import os

# imagenet images are 224x224 so we resize our custom data to 224
TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TESTING_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

client_name = cmd("hostname").strip()

DATA_PATH = "/root/data"
# DATA_PATH = "compressed_images_wheat"
TRAIN_DATA_PATH = ""
TEST_DATA_PATH = ""

try:
    if Path(DATA_PATH).exists():
        # DATA_PATH = PATHS[0]
        TRAIN_DATA_PATH = f"{DATA_PATH}/{client_name}_train.csv"
        TEST_DATA_PATH = f"{DATA_PATH}/{client_name}_test.csv"
        SERVER_TEST_DATA_PATH = f"{DATA_PATH}/test.csv"

    # elif Path(PATHS[1]).exists():
    #     DATA_PATH = PATHS[1]
    #     TRAIN_DATA_PATH = f"{DATA_PATH}/train.csv"
    #     TEST_DATA_PATH = f"{DATA_PATH}/test.csv"
except Exception as e:
    print("\n.csv data files not found: ", e, end="\n\n")

if Path(TRAIN_DATA_PATH).exists():
    DATASET = WheatImgDataset(data_file=TRAIN_DATA_PATH, transform=TRANSFORM)

    size = len(DATASET)

    train_size = int(0.8 * size)

    def split_data(dataset, train_percentage: float = 0.8):
        size = len(dataset)
        train_size = int(train_percentage * size)
        t, v = random_split(
            dataset,
            [train_size, (size - train_size)],
            generator=torch.Generator().manual_seed(33),
        )
        return t, v

    TRAINING_DATA, _ = split_data(DATASET)

    CLASSES = DATASET.classes.values()
    LABELS_MAP = {i: c for i, c in enumerate(CLASSES)}

    TRAIN_SAMPLER = oversampler(
        data_path=TRAIN_DATA_PATH, subset_indices=TRAINING_DATA.indices
    )

    DATA_SUMMARY = data_summary(TRAIN_DATA_PATH).get("size_per_class")

else:
    pass


TESTING_DATA = WheatImgDataset(
    data_file=SERVER_TEST_DATA_PATH, transform=TESTING_TRANSFORM
)


# #######################################################
# print(f"Train size: {len(TRAINING_DATA)}")
# print(f"Val size: {len(VALIDATION_DATA)}")
# print(f"Test size: {len(TESTING_DATA)}")
# print(f"First train index: {TRAINING_DATA.indices[0]}")
# print(f"First val index: {VALIDATION_DATA.indices[0]}")
# #######################################################


def data_loader(
    data,
    device: Literal["cuda", "cpu"],
    batch_size: int,
    sampler=None,
    num_workers: int = 4,
    collate_fn=None,
):
    pin_mem = False
    if device == "cuda":
        pin_mem = True
    loader = DataLoader(
        data,
        num_workers=num_workers,
        pin_memory=pin_mem,
        batch_size=batch_size,
        shuffle=(False if sampler else True),
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return loader
