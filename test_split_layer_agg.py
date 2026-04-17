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
)
from model_functions import test, eval_per_class, choose_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(dev)
print(f"\n{DEVICE}\n")


# MODELS_DIR = "/root/data/models"
MODELS_DIR = "models"


def find_model_paths(models_dir, keyword, count=2):
    # matches = sorted(glob(f"{models_dir}/cont-*_{keyword}_last_model.pt"))
    matches = sorted(glob(f"{models_dir}/cont-*{keyword}_last_model.pt"))
    return matches


def load_model(path: str):
    if "cifar" in path:
        out_features = 10
    elif "wheat" in path:
        out_features = 8
    model = choose_model(
        model_name="mobilenet_v3_large", freeze=0, out_features=out_features
    )
    state_dict = torch.load(path, weights_only=True, map_location=dev)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ======================================================
# borrowed from other test
# ======================================================
# def find_model_paths(models_dir, keyword, count=2):
#     matches = sorted(glob(f"{models_dir}/cont-*{keyword}_last_model.pt"))
#     return matches[:count]


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


def fedavg_aggregate(*models):
    # Convert each model state dict to a reply message
    replies = [
        make_train_reply(state_dict_to_array_record(m), num_examples=100)
        for m in models
    ]

    strategy = FedAvg()
    aggregated_arrays, _ = strategy.aggregate_train(server_round=1, replies=replies)
    if aggregated_arrays is None:
        raise RuntimeError("FedAvg aggregation returned None")

    # Use the keys from the first model (assumes all models share the same architecture)
    keys = list(models[0].keys())
    return {key: torch.from_numpy(aggregated_arrays[key].numpy()) for key in keys}


# =================================================================
# =================================================================

from collections import OrderedDict, defaultdict


def split_model(model) -> tuple[dict]:
    sd = model.state_dict()
    features = OrderedDict()
    classifier = OrderedDict()
    for k, v in sd.items():
        if k.startswith("features"):
            features[k] = v
        else:
            classifier[k] = v

    return features, classifier


def experiment_items(model_paths: list, labels_maps: list[dict]) -> list[dict]:
    exp = []
    for p in model_paths:
        labels = []
        for cont in labels_maps:
            name = cont.get("container")
            lm = cont.get("labels_map")
            if name in p:
                labels = list(lm.keys())

        model = load_model(p)
        f, c = split_model(model)

        item = {
            "model": p,
            "features_state_dict": f,
            "features": f,
            "classifier_state_dict": c,
            "classifier": c,
            "labels": labels,
        }
        exp.append(item)

    return exp


def analyze_elements(list_of_lists: list[list]) -> tuple[dict, dict]:
    # Map each element to the set of sublist indices it appears in
    element_locations = defaultdict(set)

    for i, sublist in enumerate(list_of_lists):
        # set(sublist) ignores duplicates within the same sublist
        for item in set(sublist):
            element_locations[item].add(i)

    # Classify: unique (appears in exactly 1 list) vs shared (appears in 2+)
    unique = {
        item: indices
        for item, indices in element_locations.items()
        if len(indices) == 1
    }
    shared = {
        item: sorted(indices)
        for item, indices in element_locations.items()
        if len(indices) > 1
    }

    return unique, shared


def custom_aggregation(exp_items: list[dict]):
    features = []
    classifiers = []
    cont_labels = []
    all_labels = []
    for ex in exp_items:
        f = ex.get("features_state_dict")
        l = ex.get("labels")
        c = ex.get("classifier_state_dict")
        features.append(f)
        classifiers.append(c)
        cont_labels.append(l)
        all_labels.extend(l)

    global_labels = set(all_labels)

    agg_features = fedavg_aggregate(*features)
    new_classifier = OrderedDict(
        {
            "classifier.0.weight": torch.stack(
                [c.get("classifier.0.weight") for c in classifiers]
            ).mean(dim=0),
            "classifier.0.bias": torch.stack(
                [c.get("classifier.0.bias") for c in classifiers]
            ).mean(dim=0),
            "classifier.3.weight": torch.zeros_like(
                classifiers[0].get("classifier.3.weight")
            ),
            "classifier.3.bias": torch.zeros_like(
                classifiers[0].get("classifier.3.bias")
            ),
        }
    )

    unique_labels, shared_labels = analyze_elements(cont_labels)

    # print(f"\n{unique_labels = }")
    # print(f"\n{shared_labels = }")

    for i in global_labels:
        if i in unique_labels:
            index= unique_labels.get(i).pop()
            target_bias = classifiers[index].get("classifier.3.bias")
            target_weight = classifiers[index].get("classifier.3.weight")
            new_classifier["classifier.3.bias"] = target_bias
            new_classifier["classifier.3.weight"] = target_weight
        if i in shared_labels:
            indecies = shared_labels.get(i)
            target_biases = [
                classifiers[idx].get("classifier.3.bias") for idx in indecies
            ]
            target_weights = [
                classifiers[idx].get("classifier.3.weight") for idx in indecies
            ]
            new_classifier["classifier.3.bias"] = torch.stack(target_biases).mean(dim=0)
            new_classifier["classifier.3.weight"] = torch.stack(target_weights).mean(
                dim=0
            )

    agg_state_dict = OrderedDict()
    agg_state_dict.update(agg_features)
    agg_state_dict.update(new_classifier)

    return agg_state_dict


# wheat_paths = find_model_paths(MODELS_DIR, "wheat")
cifar10_paths = find_model_paths(MODELS_DIR, "14*_cifar10")

labels_maps = [
    {
        "container": "cont-141",
        "labels_map": {0: "airplane", 3: "cat", 5: "dog", 6: "frog", 7: "horse"},
    },
    {
        "container": "cont-142",
        "labels_map": {1: "automobile", 2: "bird", 4: "deer", 8: "ship", 9: "truck"},
    },
]


dataset = "cifar10"
print(f"\n{'='*50}")
print(f"Dataset: {dataset.upper()}")
print(f" Models: {[os.path.basename(p) for p in cifar10_paths]}")
print(f"{'='*50}")


x_items = experiment_items(cifar10_paths, labels_maps)
agg_state_dict = custom_aggregation(x_items)

if dataset == "cifar10":
    test_loader = cifar_loader(CIFAR10_TEST, 128)
    out_features = 10
    labels_map = CIFAR10_LABELS_MAP
elif dataset == "wheat":
    test_loader = wheat_loader(WHEAT_TEST, dev, 128)
    out_features = 8
    labels_map = WHEAT_TEST.classes

agg_model = choose_model("mobilenet_v3_large", 0, out_features).to(DEVICE)
agg_model.load_state_dict(agg_state_dict)
print("Aggregation successful!")

loss_fn = torch.nn.CrossEntropyLoss()
eval_per_class(test_loader, agg_model, out_features, labels_map)
acc, loss = test(agg_model, test_loader, loss_fn)
print(f"\nGeneral Accuracy: {(acc * 100):.2f}% | Average Loss: {loss:.7f}")
