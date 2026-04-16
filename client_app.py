import torch
from torch import nn
from torch.optim import lr_scheduler
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.app import ConfigRecord
import time
import numpy as np
import json
import os
from utils import end_of_training_msg, pick_mixer, cmd
from model_functions import (
    train as train_fn,
    test as test_fn,
    choose_model,
    eval_per_class,
    EarlyStop,
)

from wheat_data_utils import WheatImgDataset


client = ClientApp()

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEV)


def modify_weights(
    out_features: int, selected_labels: list, weights: torch.Tensor
) -> torch.Tensor:
    """
    assign a weight of 0 for classes meant to be ignored during training.

    :param out_features: number of output features in the NN
    :type out_features: int
    :param selected_labels: labels of nodes to be trained
    :type selected_labels: list
    :param weights: class weights for local data
    :type weights: torch.Tensor
    :return: class weights for model training on this client
    :rtype: Tensor
    """

    new_weights = np.ones(out_features)
    c = 0
    for i, _ in enumerate(new_weights):
        if i in selected_labels:
            new_weights[i] = weights.tolist()[c]
            c += 1

    return torch.tensor(new_weights).float()


def generate_local_labels_map(local_classes: list[str], local_labels: list[int]):
    """
    create a labels map for the local classes such that the labels are identical to the global model

    :param local_classes: local data classes
    :type local_classes: list[str]
    :param local_labels: labels of local classes in the global model
    :type local_labels: list[int]
    :return: local labels map
    :rtype: dict[int, str]
    """
    l_m = {i: c for i, c in zip(local_labels, local_classes)}
    return l_m


@client.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    node_name = cmd("hostname").strip()

    train_acc_data = []
    val_acc_data = []
    train_loss_data = []
    val_loss_data = []
    train_times = []
    passed_epochs = []
    f_lrs = []
    c_lrs = []
    # model params
    print(f"######################################################")
    print(f"\n Loading Params \n")
    print(f"{msg.content.get("config") = }\n")
    print(f"######################################################")

    server_config = msg.content["config"]

    model_name = server_config.get("model-name", context.run_config["model-name"])
    freeze = server_config.get("freeze", context.run_config["freeze"])
    batch_size = server_config.get("batch-size", context.run_config["batch-size"])
    use_sampler = server_config.get("use-sampler", context.run_config["use-sampler"])
    num_workers = server_config.get("num-workers", context.run_config["num-workers"])
    if DEV == "cpu":
        num_workers = 0
    features_lr = server_config.get("features-lr", context.run_config["features-lr"])
    classifier_lr = server_config.get(
        "classifier-lr", context.run_config["classifier-lr"]
    )
    weight_decay = server_config.get("weight-decay", context.run_config["weight-decay"])
    sch_patience = server_config.get("sch-patience", context.run_config["sch-patience"])
    use_weights = server_config.get("use-weights", context.run_config["use-weights"])
    epochs = server_config.get("local-epochs", context.run_config["local-epochs"])
    dataset_name = server_config.get("dataset-name", context.run_config["dataset-name"])
    mixer = server_config.get("mixer", context.run_config["mixer"])
    out_features = server_config.get("out-features")
    labels = server_config.get("labels")
    proximal_mu = server_config.get("proximal-mu")

    if dataset_name == "wheat":
        from wheat_data_utils import get_class_weights
        from wheat_data_prep import (
            DATASET as wheat_dataset,
            TRAIN_DATA_PATH,
            TEST_DATA_PATH,
            TESTING_TRANSFORM,
            split_data,
            data_loader,
            TRAIN_SAMPLER,
            CLASSES as wheat_classes,
        )

        local_classes = list(wheat_classes)

    elif dataset_name == "cifar10":
        from cifar10_data_prep import (
            CIFAR10_CLASSES,
            CIFAR10_LABELS_MAP,
            CIFAR10_TRAIN,
            CIFAR10_VAL,
            CIFAR10_TEST,
            loader as cifar_loader,
        )

        local_classes = CIFAR10_CLASSES
        local_labels_map = CIFAR10_LABELS_MAP
        trainloader = cifar_loader(CIFAR10_TRAIN, batch_size)
        valloader = cifar_loader(CIFAR10_VAL, batch_size)
        testloader = cifar_loader(CIFAR10_TEST, 128)

        mixer = ""
        weights = None

    # check if this is a prep phase, return classes if True
    prep_phase = server_config.get("prep-phase")
    if prep_phase:
        node_id = context.node_id
        prep_conf = ConfigRecord(
            {"local-classes": local_classes, "node-name": node_name, "node-id": node_id}
        )
        content = RecordDict({"config": prep_conf})
        print("\nPreparation Phase complete\n")
        return Message(content=content, reply_to=msg)

    if labels:
        print("\n--> Local labels: ", labels)
        # print("--> Weights: ", class_weights, end="\n\n")
        test_conf = ConfigRecord({"node-name": node_name})
        content = RecordDict({"config": test_conf})
        with open("assigned_labels.json", "w") as f:
            json.dump({"labels": labels}, f)
        return Message(content=content, reply_to=msg)
    else:
        with open("assigned_labels.json", "r") as f:
            labels = json.load(f).get("labels")

    if dataset_name == "wheat":
        local_labels_map = generate_local_labels_map(local_classes, labels)
        print(f"\n{local_labels_map = }\n")
        wheat_dataset.change_class_labels(local_labels_map)
        wheat_train, wheat_val = split_data(wheat_dataset)
        trainloader = data_loader(
            wheat_train,
            DEV,
            batch_size,
            TRAIN_SAMPLER if use_sampler else None,
            num_workers=num_workers,
        )
        valloader = data_loader(
            wheat_val,
            DEV,
            batch_size,
            num_workers=num_workers,
        )

        test_dataset = WheatImgDataset(
            data_file=TEST_DATA_PATH,
            labels_map=local_labels_map,
            transform=TESTING_TRANSFORM,
        )
        testloader = data_loader(test_dataset, DEV, 128)

        class_weights = get_class_weights(TRAIN_DATA_PATH, wheat_train.indices).to(
            DEVICE
        )
        modified_weights = modify_weights(out_features, labels, class_weights).to(
            DEVICE
        )
        print("--> Modified Weights: ", modified_weights)
        print(" ")
        weights = modified_weights if use_weights else None

    # Load the model and initialize it with the received weights
    print("\nDevice: ", DEVICE)
    print("\nChosen model: ", model_name)
    print("\nDataset: ", dataset_name.upper())
    # print("\nLocal classes: ", local_classes)
    print(f"\nLocal labels map: {local_labels_map}")
    print(f"\nOutput features: {out_features}")
    print(f"\nProximal mu: {proximal_mu}")
    print(f"\nMixer used: {mixer}")
    for _, lbls in trainloader:
        print(f"\nFirst batch labels: {lbls.unique()}\n")
        break

    model = choose_model(model_name, freeze, out_features).to(DEVICE)

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # necessary for FedProx
    global_params = [p.clone().detach().to(DEVICE) for p in model.parameters()]

    # optimizer and loss_fn

    opt_algo = torch.optim.AdamW
    optimizer = opt_algo(model.parameters(), classifier_lr, weight_decay=weight_decay)
    # for unfrozen backbone
    # if not freeze:
    #     optimizer = opt_algo(
    #         [
    #             {"params": model.features.parameters(), "lr": features_lr},
    #             {"params": model.classifier.parameters(), "lr": classifier_lr},
    #         ],
    #         weight_decay=weight_decay,
    #     )
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=sch_patience)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # commence training loop
    mixer = pick_mixer(mixer, out_features)

    current_round = server_config.get("current-round")

    stopper = EarlyStop(3, 0.5)
    try:
        for e in range(epochs):
            print(
                f"\nEpoch {e+1}/{epochs} | Round {current_round}\n-------------------------------"
            )

            # f_lr = optimizer.param_groups[0]["lr"]
            # c_lr = optimizer.param_groups[1]["lr"]
            c_lr = optimizer.param_groups[0]["lr"]
            # print(f"Features learning rate: {f_lr}")
            # print(f"Classifier learning rate: {c_lr}\n")

            t0 = time.perf_counter()
            print("Training commencing...")
            train_acc, train_loss = train_fn(
                model,
                trainloader,
                labels,
                optimizer,
                loss_fn,
                global_params,
                mixer,
            )

            print("validation...")
            # ignore_lbls = ignored_labels(out_features, labels)
            # val_acc, val_loss = 0, 0
            val_acc, val_loss = test_fn(model, valloader, loss_fn)

            stopper.record(e + 1, train_loss, val_loss)
            print(f"\n{stopper.values = }")
            print(f"{stopper.current_delta = }")
            print(f"{stopper.delta_slope() = }")
            print(f"{stopper.early_stopper() = }")
            print(f"{stopper.counter = }\n")

            print("Gathering data...")
            train_acc_data.append(train_acc)
            train_loss_data.append(train_loss)
            val_acc_data.append(val_acc)
            val_loss_data.append(val_loss)
            t1 = time.perf_counter() - t0
            train_times.append(t1)
            passed_epochs.append(e + 1)
            # f_lrs.append(f_lr)
            # c_lrs.append(c_lr)
            f_lrs.append(c_lr)
            c_lrs.append(c_lr)

            # scheduler.step(val_loss)

            print(
                f"Training metrics:\n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n"
            )
            print(
                f"Validation metrics:\n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n"
            )
            print("End of epoch.\n")
    except Exception as e:
        print("Training crashed: ", e)
        raise e

    end_of_training_msg(sum(train_times))
    local_model_path = f"/root/data/models/{node_name}_{dataset_name}_last_model.pt"
    print(f"\nSaving local model...")
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    torch.save(model.state_dict(), local_model_path)

    print("\n---------------------------------------------------------------")
    print("\nGeneral evaluation:\n")
    test_acc, test_loss = test_fn(model, testloader, loss_fn)
    print(
        f"Testing metrics:\n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    print("\nPer-class local evaluation:\n")
    eval_per_class(testloader, model, out_features, local_labels_map)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "classifier-lr": c_lrs,
        "features-lr": f_lrs,
        "train-acc": train_acc_data,
        "train-loss": train_loss_data,
        "val-acc": val_acc_data,
        "val-loss": val_loss_data,
        "train-time": train_times,
        "epoch": passed_epochs,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    config_record = ConfigRecord(
        {
            "client-name": node_name,
            "local-classes": local_classes,
        }
    )
    content = RecordDict(
        {"arrays": model_record, "metrics": metric_record, "configs": config_record}
    )

    return Message(content=content, reply_to=msg)


# @client.evaluate()
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""
#     IMG_C = context.run_config["img_c"]
#     IMG_H = context.run_config["img_h"]
#     OUTPUT_CHANNELS = literal_eval(context.run_config["out_channels"])
#     KERNEL_SIZE = context.run_config["kernel_size"]
#     CLASSES = literal_eval(context.run_config["classes"])
#     DATASET_ID = context.run_config["dataset_id"]

#     # Load the model and initialize it with the received weights
#     model = CNN(
#         in_channels=IMG_C,
#         out_channels=OUTPUT_CHANNELS,
#         kernel_size=KERNEL_SIZE,
#         out_features=len(CLASSES),
#         img_h=IMG_H,
#     ).to(DEVICE)
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     batch_size = context.run_config["batch-size"]

#     dev = "cuda:0" if torch.cuda.is_available() else "cpu"
#     validation_loader = data_loader(
#         VALIDATION_DATA,
#         dev,
#         batch_size,
#         num_workers=num_workers,
#     )
#     # Call the evaluation function
#     eval_loss, eval_acc = test_fn(
#         model,
#         valloader,
#         DEVICE,
#     )

#     # Construct and return reply Message
#     metrics = {
#         "eval_loss": eval_loss,
#         "eval_acc": eval_acc,
#         "num-examples": len(valloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)
