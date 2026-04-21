"""
under construction/testing
"""

import torch
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from flwr.app import (
    ArrayRecord,
    RecordDict,
    ConfigRecord,
    MetricRecord,
    Array,
    Message,
    MessageType,
)
from flwr.serverapp.strategy import FedAvg


def state_dict_to_array_record(state_dict) -> ArrayRecord:
    return ArrayRecord({key: Array(val.cpu()) for key, val in state_dict.items()})


def analyze_elements(
    list_of_lists: list[list],
) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    """
    look for unique and shared elements in a list of lists.
    return dict of elements and the indices of lists where they occur

    :param list_of_lists: list of lists, same type of elements
    :type list_of_lists: list[list]
    :return: shared and unique elements and where they occur
    :rtype: tuple[dict, dict]
    """
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


def split_state_dict(state_dict) -> tuple[dict]:
    features = OrderedDict()
    classifier = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("features"):
            features[k] = v
        else:
            classifier[k] = v

    return features, classifier


def custom_classifier_aggregation(clients_data: list[dict], global_labels: list):
    """
    custom aggregation algorithm.
    param items format:
    {
        int:{
                "labels": list[int],
                "classifier_state_dict": torch_state_dict,
            }
    }

    :param exp_items: Description
    :type exp_items: list[dict]
    :return: Description
    :rtype: OrderedDict
    """
    classifiers = []
    cont_labels = []
    for item in clients_data:
        l = item.get("labels")
        c = item.get("classifier_state_dict")
        classifiers.append(c)
        cont_labels.append(l)

    w0 = torch.stack([c.get("classifier.0.weight") for c in classifiers]).mean(dim=0)
    b0 = torch.stack([c.get("classifier.0.bias") for c in classifiers]).mean(dim=0)
    w3 = torch.zeros_like(
        classifiers[0].get("classifier.3.weight")
    )  # shape: tensor([out_features, in_features])
    b3 = torch.zeros_like(
        classifiers[0].get("classifier.3.bias")
    )  # shape: tensor([out_features])

    unique_labels, shared_labels = analyze_elements(cont_labels)

    for label in global_labels:
        if label in unique_labels:
            client_idx = next(iter(unique_labels.get(label)))
            target_bias = classifiers[client_idx].get("classifier.3.bias")
            target_weight = classifiers[client_idx].get("classifier.3.weight")
            b3[label] = target_bias[label]
            w3[label] = target_weight[label]

        if label in shared_labels:
            client_indices = shared_labels.get(label)
            target_biases = [
                classifiers[idx].get("classifier.3.bias")[label]
                for idx in client_indices
            ]
            target_weights = [
                classifiers[idx].get("classifier.3.weight")[label]
                for idx in client_indices
            ]
            b3[label] = torch.stack(target_biases).mean(dim=0)
            w3[label] = torch.stack(target_weights).mean(dim=0)
    new_classifier = OrderedDict(
        {
            "classifier.0.weight": w0,
            "classifier.0.bias": b0,
            "classifier.3.weight": w3,
            "classifier.3.bias": b3,
        }
    )
    return new_classifier


def merge_state_dicts(features, classifier):
    agg_state_dict = OrderedDict()
    agg_state_dict.update(features)
    agg_state_dict.update(classifier)
    return agg_state_dict


def perform_custom_aggregation(
    train_replies: Iterable[Message], agg_arrays: ArrayRecord
) -> ArrayRecord:
    agg_features_state_dict, _ = split_state_dict(agg_arrays.to_torch_state_dict())
    clients_data = []
    global_labels = set()
    for msg in train_replies:
        if not msg.has_content():
            continue
        labels = msg.content.get("configs").get("local-labels")
        local_state_dict = msg.content.get("arrays").to_torch_state_dict()
        _, classifier = split_state_dict(local_state_dict)
        item = {
            "labels": labels,
            "classifier_state_dict": classifier,
        }
        clients_data.append(item)
        global_labels.update(labels)

    agg_classifier_state_dict = custom_classifier_aggregation(
        clients_data, global_labels
    )
    agg_model_state_dict = merge_state_dicts(
        agg_features_state_dict, agg_classifier_state_dict
    )
    agg_arrays = state_dict_to_array_record(agg_model_state_dict)
    return agg_arrays
