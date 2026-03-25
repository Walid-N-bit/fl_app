from wheat_data_utils import WheatImgDataset, oversampler
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
from typing import Literal
from torch.utils.data import DataLoader, random_split
import torch
from utils import cmd

# imagenet images are 224x224 so we resize our custom data to 224
TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

DATA_PATH = "/root/data"
client_name = cmd("hostname").strip()
TRAIN_DATA_PATH = f"{DATA_PATH}/{client_name}_train.csv"
TEST_DATA_PATH = f"{DATA_PATH}/{client_name}_test.csv"


DATASET = WheatImgDataset(data_file=TRAIN_DATA_PATH, transform=TRANSFORM)

size = len(DATASET)

train_size = int(0.8 * size)

TRAINING_DATA, VALIDATION_DATA = random_split(
    DATASET,
    [train_size, (size - train_size)],
    generator=torch.Generator().manual_seed(33),
)


TESTING_DATA = WheatImgDataset(data_file=TEST_DATA_PATH, transform=TRANSFORM)

# #######################################################
# print(f"Train size: {len(TRAINING_DATA)}")
# print(f"Val size: {len(VALIDATION_DATA)}")
# print(f"Test size: {len(TESTING_DATA)}")
# print(f"First train index: {TRAINING_DATA.indices[0]}")
# print(f"First val index: {VALIDATION_DATA.indices[0]}")
# #######################################################

CLASSES = DATASET.classes.values()
LABELS_MAP = {i: c for i, c in enumerate(CLASSES)}


def pick_mixer(name: str, num_classes: int):
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


TRAIN_SAMPLER = oversampler(
    data_path=TRAIN_DATA_PATH, subset_indices=TRAINING_DATA.indices
)


def data_loader(
    data,
    device: Literal["cuda", "cpu"],
    batch_size: int,
    sampler=None,
    num_workers: int = 4,
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
    )
    return loader
