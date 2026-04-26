import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import ShardPartitioner
from flwr_datasets.utils import divide_dataset
from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import cmd

CLIENT_NAME = cmd("hostname").strip()
if "cont" in CLIENT_NAME:
    ID = int(CLIENT_NAME[-1])
else:
    ID = None


# partition = fds.load_partition(0, "train")


def cifar10_partitioner(num_clients, num_classes):
    return ShardPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        num_shards_per_partition=num_classes,
    )


NUM_CLIENTS = 3
NUM_SHARDS_PER_CLIENT = 4

partitioner = cifar10_partitioner(NUM_CLIENTS, NUM_SHARDS_PER_CLIENT)


def cifar10_fds(partitioner):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    return fds


fds = cifar10_fds(partitioner)
if ID is not None and ID > 0:
    local_dataset = fds.load_partition(ID - 1, "train")
else:

    local_dataset = load_dataset("cifar10", split="train")

# local_dataset = load_dataset("cifar10", split="train")

# transforms = ToTensor()
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


def classes_list(partition):
    unique_labels = sorted(set(partition["label"]))
    all_names = partition.features["label"].names
    return [all_names[i] for i in unique_labels]


CIFAR10_CLASSES = classes_list(local_dataset)
unique_labels = sorted(set(local_dataset["label"]))
all_names = local_dataset.features["label"].names
CIFAR10_LABELS_MAP = {i: all_names[i] for i in unique_labels}


def apply_transforms(batch):
    batch["img"] = [TRANSFORM(img) for img in batch["img"]]
    return batch


local_dataset = local_dataset.with_transform(apply_transforms)


class DSWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return (item["img"], item["label"])


train, valid, test = divide_dataset(local_dataset, [0.6, 0.2, 0.2])
CIFAR10_TRAIN = DSWrapper(train)
CIFAR10_VAL = DSWrapper(valid)
CIFAR10_TEST = DSWrapper(test)


def loader(dataset, batch_size: int):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    pin_mem = False if dev == "cpu" else True
    return DataLoader(dataset, pin_memory=pin_mem, batch_size=batch_size)
