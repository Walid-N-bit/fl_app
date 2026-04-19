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


# def make_train_reply(arrays: ArrayRecord, num_examples: int):
#     content = RecordDict(
#         records={
#             "parameters": arrays,
#             "metrics": MetricRecord({"num-examples": num_examples}),
#         }
#     )
#     return Message(content=content, dst_node_id=0, message_type=MessageType.TRAIN)


# def fedavg_aggregate(*models):
#     """
#     the standard FedAvg aggregation method

#     :param models: client model state dicts to be aggregated
#     :return: aggregated global model state dict
#     :rtype: dict[Any, Tensor]
#     """
#     # Convert each model state dict to a reply message
#     replies = [
#         make_train_reply(state_dict_to_array_record(m), num_examples=100)
#         for m in models
#     ]

#     strategy = FedAvg()
#     aggregated_arrays, _ = strategy.aggregate_train(server_round=1, replies=replies)
#     if aggregated_arrays is None:
#         raise RuntimeError("FedAvg aggregation returned None")

#     # Use the keys from the first model (assumes all models share the same architecture)
#     keys = list(models[0].keys())
#     return {key: torch.from_numpy(aggregated_arrays[key].numpy()) for key in keys}


def analyze_elements(list_of_lists: list[list]) -> tuple[dict[dict], dict[list]]:
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

    for i in global_labels:
        if i in unique_labels:
            index = unique_labels.get(i).pop()
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
