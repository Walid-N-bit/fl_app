import os
import torch
import numpy as np
from glob import glob
from flwr.app import ArrayRecord, Array, Message, RecordDict, MetricRecord, MessageType
from flwr.serverapp.strategy import FedAvg
from cifar10_data_prep import CIFAR10_TEST, loader as cifar_loader, CIFAR10_LABELS_MAP
from wheat_data_prep import (
    TESTING_DATA as WHEAT_TEST,
    data_loader as wheat_loader,
    LABELS_MAP as WHEAT_LABELS_MAP,
)
from model_functions import test, eval_per_class, choose_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(dev)
print(f"\n{DEVICE}\n")


def find_model_paths(models_dir, keyword, count=2):
    matches = sorted(glob(f"{models_dir}/cont-*_{keyword}_last_model.pt"))
    return matches[:count]


EXPERIMENTS = [
    {
        "model_paths": find_model_paths("models", "cifar10"),
        "dataset": "cifar10",
        "out_features": 10,
        "labels_map": CIFAR10_LABELS_MAP,
    },
    {
        "model_paths": find_model_paths("models", "wheat"),
        "dataset": "wheat",
        "out_features": 8,
        "labels_map": WHEAT_LABELS_MAP,
    },
]


def state_dict_to_array_record(state_dict):
    return ArrayRecord({key: Array(val.cpu()) for key, val in state_dict.items()})


def make_train_reply(arrays: ArrayRecord, num_examples: int):
    content = RecordDict(
        records={
            "parameters": arrays,
            "metrics": MetricRecord({"num-examples": num_examples}),
        }
    )
    return Message(content=content, dst_node_id=0, message_type=MessageType.TRAIN)


def fedavg_aggregate(model_1, model_2):
    arrays_1 = state_dict_to_array_record(model_1)
    arrays_2 = state_dict_to_array_record(model_2)
    replies = [
        make_train_reply(arrays_1, num_examples=100),
        make_train_reply(arrays_2, num_examples=100),
    ]
    strategy = FedAvg()
    aggregated_arrays, _ = strategy.aggregate_train(server_round=1, replies=replies)
    if aggregated_arrays is None:
        raise RuntimeError("FedAvg aggregation returned None")
    keys = list(model_1.keys())
    return {key: torch.from_numpy(aggregated_arrays[key].numpy()) for key in keys}


loss_fn = torch.nn.CrossEntropyLoss()

for exp in EXPERIMENTS:
    dataset = exp["dataset"]
    out_features = exp["out_features"]
    labels_map = exp["labels_map"]
    model_paths = exp["model_paths"]

    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing or len(model_paths) < 2:
        print(f"\nSkipping {dataset.upper()}: not enough model files found in models/")
        for p in missing:
            print(f"  - {p}")
        continue

    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset.upper()}")
    print(f"  Models:  {[os.path.basename(p) for p in model_paths]}")
    print(f"{'='*50}")

    model_1 = torch.load(model_paths[0], weights_only=True)
    model_2 = torch.load(model_paths[1], weights_only=True)

    agg_state_dict = fedavg_aggregate(model_1, model_2)

    agg_model = choose_model("mobilenet_v3_large", 0, out_features).to(DEVICE)
    agg_model.load_state_dict(agg_state_dict)
    print("Aggregation successful!")

    if dataset == "cifar10":
        test_loader = cifar_loader(CIFAR10_TEST, 128)
    else:
        test_loader = wheat_loader(WHEAT_TEST, dev, 128)

    eval_per_class(test_loader, agg_model, out_features, labels_map)
    acc, loss = test(agg_model, test_loader, loss_fn)
    print(f"\nGeneral Accuracy: {(acc * 100):.2f}% | Average Loss: {loss:.7f}")
