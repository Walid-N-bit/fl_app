import torch
from torch import nn
from torch.optim import lr_scheduler
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from ast import literal_eval
import time
from utils import end_of_training_msg, pick_mixer
from model_functions import train as train_fn, test as test_fn, choose_model
from wheat_data_prep import (
    TRAINING_DATA as wheat_train,
    VALIDATION_DATA as wheat_val,
    data_loader,
    TRAIN_SAMPLER,
    CLASSES as wheat_classes,
)
from wheat_data_utils import get_class_weights
from cifar10_data_prep import (
    CIFAR10_CLASSES,
    CIFAR10_TRAIN,
    CIFAR10_VAL,
    CIFAR10_TEST,
    loader,
)

client = ClientApp()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@client.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    train_times = []
    passed_epochs = []
    f_lrs = []
    c_lrs = []
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model params
    server_config = msg.content["config"]

    model_name = server_config.get("model-name", context.run_config["model-name"])
    freeze = server_config.get("freeze", context.run_config["freeze"])
    batch_size = server_config.get("batch-size", context.run_config["batch-size"])
    use_sampler = server_config.get("use-sampler", context.run_config["use-sampler"])
    num_workers = server_config.get("num-workers", context.run_config["num-workers"])
    if dev == "cpu":
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

    if dataset_name == "wheat":
        local_classes = wheat_classes
        trainloader = data_loader(
            wheat_train,
            dev,
            batch_size,
            TRAIN_SAMPLER if use_sampler else None,
            num_workers=num_workers,
        )
        valloader = data_loader(
            wheat_val,
            dev,
            batch_size,
            num_workers=num_workers,
        )
        class_weights = get_class_weights(
            "compressed_images_wheat/train.csv", wheat_train.indices
        ).to(DEVICE)

    elif dataset_name == "cifar10":
        local_classes = CIFAR10_CLASSES
        trainloader = loader(CIFAR10_TRAIN, batch_size)
        valloader = loader(CIFAR10_VAL, batch_size)
        mixer = ""

    # Load the model and initialize it with the received weights

    print("\nDevice: ", DEVICE)
    print("\nChosen model: ", model_name)
    print("\nDataset: ", dataset_name.upper())
    print("\nClasses: ", local_classes)
    print("\nDataset: ", dataset_name.upper())

    model = choose_model(model_name, freeze, len(local_classes)).to(DEVICE)

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data

    # optimizer and loss_fn

    opt_algo = torch.optim.AdamW
    optimizer = opt_algo(
        model.classifier.parameters(), classifier_lr, weight_decay=weight_decay
    )
    # for unfrozen backbone
    if not freeze:
        optimizer = opt_algo(
            [
                {"params": model.features.parameters(), "lr": features_lr},
                {"params": model.classifier.parameters(), "lr": classifier_lr},
            ],
            weight_decay=weight_decay,
        )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=sch_patience)

    loss_fn = nn.CrossEntropyLoss(weight=(class_weights if use_weights else None))

    # commence training loop
    mixer = pick_mixer(mixer, local_classes)

    try:
        for e in range(epochs):
            print(f"Epoch {e+1}\n-------------------------------")

            f_lr = optimizer.param_groups[0]["lr"]
            c_lr = optimizer.param_groups[1]["lr"]
            print(f"Features learning rate: {f_lr}")
            print(f"Classifier learning rate: {c_lr}\n")

            t0 = time.perf_counter()
            print("Training commencing...")
            train_acc, train_loss = train_fn(
                model, trainloader, optimizer, loss_fn, mixer
            )
            print("validation...")
            val_acc, val_loss = test_fn(model, valloader, loss_fn)
            print("Gathering data...")
            t1 = time.perf_counter() - t0
            train_times.append(t1)
            passed_epochs.append(e + 1)
            f_lrs.append(f_lr)
            c_lrs.append(c_lr)

            scheduler.step(val_loss)

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

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "classifier_lr": c_lrs,
        "features_lr": f_lrs,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_times": train_times,
        "epochs": passed_epochs,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
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
