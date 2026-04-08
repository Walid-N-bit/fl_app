import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights,
)
from wheat_data_utils import get_class_weights
from typing import Literal

from CustomClasses import ConvolutionalNeuralNetwork as CNN

global NET
NET = models.MobileNetV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_model(
    model_name: Literal[
        "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "cnn"
    ],
    freeze: bool | Literal[0, 1],
    out_features: int,
):
    match model_name:
        case "mobilenet_v3_small":
            model = models.mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
            ).to(DEVICE)
            if freeze:
                for param in model.parameters():
                    param.requires_grad = False

            model.classifier[2] = nn.Dropout(p=0.3, inplace=True)
            model.classifier[3] = nn.Linear(in_features=1024, out_features=out_features)
            model.classifier.insert(0, nn.Dropout(p=0.3, inplace=True))

        case "mobilenet_v3_large":
            model = models.mobilenet_v3_large(MobileNet_V3_Large_Weights.DEFAULT).to(
                DEVICE
            )
            if freeze:
                for param in model.parameters():
                    param.requires_grad = False

            model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
            model.classifier[3] = nn.Linear(in_features=1280, out_features=out_features)
            # model.classifier.insert(0, nn.Dropout(p=0.3, inplace=True))

        case "efficientnet_b0":
            model = models.efficientnet_b0(EfficientNet_B0_Weights.DEFAULT).to(DEVICE)
            if freeze:
                for param in model.parameters():
                    param.requires_grad = False
            model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
            model.classifier[1] = nn.Linear(
                in_features=1280, out_features=out_features, bias=True
            )
            NET = models.EfficientNet
        case "cnn":
            model = CNN(out_features).to(DEVICE)
            NET = CNN

    return model


def train(
    model: NET,
    trainloader: DataLoader,
    optimizer,
    loss_func,
    mixer=None,
    disp_log: bool = True,
    max_grad_norm: float = 1.0,
):
    """Train the model on the training set."""

    size = len(trainloader.dataset)
    num_batches = len(trainloader)
    train_acc, train_loss = 0, 0
    model.train()

    disp_window = []
    disp_window_size = 20

    for batch, pair in enumerate(trainloader):

        (X, y) = pair
        images = X.to(DEVICE)
        labels_hard = y.to(DEVICE)
        if mixer:
            images, labels = mixer(images, labels_hard)
        else:
            labels = labels_hard

        predictions = model(images)

        # if (batch % 100 == 0) and disp_log:
        #     print(f"{labels.shape = }\n{predictions.shape = }")
        #     print(f"{labels.dtype = }\n{predictions.dtype = }")
        #     print(f"{images.dtype = }\n")
        #     print("")

        loss = loss_func(predictions, labels)

        pred_labels = predictions.argmax(1)

        if labels.ndim > 1:
            target_labels = labels.argmax(1)
        else:
            target_labels = labels

        train_acc += (
            (pred_labels == target_labels).type(torch.float).sum().item()
        )  # here all correct preds are summed
        train_loss += loss.item()  # this returns the average loss over the batch

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # to prevent gradient explosion
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        ########

        # average loss over a certain number of batches
        disp_window.append(loss.item())
        if len(disp_window) >= disp_window_size:
            disp_window.pop(0)

        if batch % 100 == 0 and disp_log:
            # print("a sample of labels: ", labels_hard.unique())
            avg_loss = sum(disp_window) / len(disp_window)
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"current loss: {loss:>7.6f}  [{current:>5d}/{size:>5d}]")
            print(f"average loss: {avg_loss:>7.6f}  [{current:>5d}/{size:>5d}]")

    train_acc = train_acc / size
    train_loss = train_loss / num_batches

    return train_acc, train_loss


def test(model: NET, testloader: DataLoader, loss_func, ignore_labels: list = []):
    """Validate the model on the test set."""
    size = len(testloader.dataset)
    num_batches = len(testloader)
    model.eval()
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.no_grad():
        batch_count = 0
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # if ignore_labels:
            #     mask = torch.isin(labels, torch.tensor(ignore_labels).to(DEVICE))
            #     # labels[mask] = -100
            #     if mask.all():
            #         continue

            predictions = model(images)
            loss = loss_func(predictions, labels)
            test_loss += loss.item()
            test_acc += (predictions.argmax(1) == labels).type(torch.float).sum().item()
            # if batch_count % 50 == 0:
            #     print("\nA sample of labels: ", labels.unique())
            #     print("")
            # batch_count += 1

    # test_loss /= batch_count
    test_loss /= num_batches
    test_acc /= size

    return test_acc, test_loss


def eval_per_class(testloader, model, out_features: int, labels_map: dict):

    actual_values = []
    pred_values = []
    # prepare to count predictions for each class
    global_labels_map = {
        i: c for i, c in enumerate(["Unknown-Class" for _ in range(out_features)])
    }
    print(f"\nLocal labels map: {labels_map}")
    print(f"\nTemplate labels map: {global_labels_map}")

    if len(labels_map) == out_features:
        global_labels_map = labels_map
    else:
        for i in global_labels_map:
            if i in labels_map:
                class_name = labels_map.get(i)
                global_labels_map[i] = class_name

    classes = list(global_labels_map.values())
    print(f"\nGlobal labels map: {global_labels_map}")
    print(f"\nClasses: {classes}\n")
    correct_pred = {classname: 0.0 for classname in classes}
    total_pred = {classname: 0.0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            actual_values.extend(labels)
            pred_values.extend(predictions)
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            print(f"Total predictions for {classname} = {total_pred[classname]}")
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

    return actual_values, pred_values


class EarlyStop:

    def __init__(self, tolerance: int = 1, min_delta: float = 0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.values = []

    def delta(self, training_loss, validation_loss) -> float:
        self.current_delta = (validation_loss - training_loss) / (validation_loss)

    def record(self, current_epoch):
        if len(self.values) >= 2:
            self.values.pop(0)
        self.values.append((current_epoch, self.current_delta))

    def delta_slope(self, epoch):
        self.record(epoch)
        if len(self.values) == 2:
            e1, d1 = self.values[0]
            e2, d2 = self.values[1]
            return (d2 - d1) / (e2 - e1)
        else:
            return 0

    def early_stopper(self, training_loss, validation_loss) -> bool:
        self.delta(validation_loss, training_loss)
        if self.delta_slope() > 0:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter >= self.tolerance:
            return True
        else:
            return False
