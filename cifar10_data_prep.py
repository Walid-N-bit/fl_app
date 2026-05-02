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

# partition = fds.load_partition(0, "train")


def cifar10_partitioner(num_clients=2, num_shards=5):
    return ShardPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        num_shards_per_partition=num_shards,
        shuffle=False,
    )


class DSWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return (item["img"], item["label"])


def cifar10_fds(partitioner):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    return fds


def classes_list(partition):
    unique_labels = sorted(set(partition["label"]))
    all_names = partition.features["label"].names
    return [all_names[i] for i in unique_labels]


def apply_transforms(batch):
    batch["img"] = [TRANSFORM(img) for img in batch["img"]]
    return batch


def loader(dataset, batch_size: int):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    pin_mem = False if dev == "cpu" else True
    return DataLoader(dataset, pin_memory=pin_mem, batch_size=batch_size)


def get_local_dataset(partitioner, client_id: int = ID):
    fds = cifar10_fds(partitioner)
    if client_id is not None and client_id > 0:
        local_dataset = fds.load_partition(client_id - 1, "train")
    else:

        local_dataset = load_dataset("cifar10", split="train")
    return local_dataset


def get_cifar10_labels_map(local_dataset):
    unique_labels = sorted(set(local_dataset["label"]))
    all_names = local_dataset.features["label"].names
    labels_map = {i: all_names[i] for i in unique_labels}
    return labels_map


def get_cifar10_dataset_splits(
    num_clients=2, num_shards=5, division: list = [0.6, 0.2, 0.2]
):
    partitioner = cifar10_partitioner(num_clients, num_shards)
    local_dataset = get_local_dataset(partitioner)
    labels_map = get_cifar10_labels_map(local_dataset)
    classes_names = classes_list(local_dataset)
    local_dataset = local_dataset.with_transform(apply_transforms)
    train, valid, test = divide_dataset(local_dataset, division)

    return (
        DSWrapper(train),
        DSWrapper(valid),
        DSWrapper(test),
        labels_map,
        classes_names,
    )


# partitioner = cifar10_partitioner()


# if ID is not None and ID > 0:
#     local_dataset = fds.load_partition(ID - 1, "train")
# else:

#     local_dataset = load_dataset("cifar10", split="train")


# CIFAR10_CLASSES = classes_list(local_dataset)
# unique_labels = sorted(set(local_dataset["label"]))
# all_names = local_dataset.features["label"].names
# CIFAR10_LABELS_MAP = {i: all_names[i] for i in unique_labels}

# train, valid, test = divide_dataset(local_dataset, [0.6, 0.2, 0.2])
# CIFAR10_TRAIN = DSWrapper(train)
# CIFAR10_VAL = DSWrapper(valid)
# CIFAR10_TEST = DSWrapper(test)


test_ds = load_dataset("cifar10", split="test")
test_ds = test_ds.with_transform(apply_transforms)
CIFAR10_TEST = DSWrapper(test_ds)
# _, _, _, _, CIFAR10_CLASSES = get_cifar10_dataset_splits()
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
