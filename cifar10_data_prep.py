import torch
from flwr_datasets import FederatedDataset

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from flwr_datasets.utils import divide_dataset

fds = FederatedDataset(dataset="cifar10", partitioners={"train": 2})
partition = fds.load_partition(0, "train")
centralized_dataset = fds.load_split("test")


# transforms = ToTensor()
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


def apply_transforms(batch):
    batch["img"] = [TRANSFORM(img) for img in batch["img"]]
    return batch


partition = partition.with_transform(apply_transforms)
# Now, you can check if you didn't make any mistakes by calling partition_torch[0]

CIFAR10_CLASSES = partition.features["label"].names
CIFAR10_LABELS_MAP = {i: c for i, c in enumerate(CIFAR10_CLASSES)}


train, valid, test = divide_dataset(partition, [0.6, 0.2, 0.2])


class DSWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return (item["img"], item["label"])


CIFAR10_TRAIN = DSWrapper(train)
CIFAR10_VAL = DSWrapper(valid)
CIFAR10_TEST = DSWrapper(test)


def loader(dataset, batch_size: int):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    pin_mem = False if dev == "cpu" else True
    return DataLoader(dataset, pin_memory=pin_mem, batch_size=batch_size)


# TRAIN_LOADER = DataLoader(train, pin_memory=pin_mem, batch_size=32)
# VAL_LOADER = DataLoader(valid, pin_memory=pin_mem, batch_size=32)
# TEST_LOADER = DataLoader(test, pin_memory=pin_mem, batch_size=32)
