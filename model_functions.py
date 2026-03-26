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
):
    """Train the model on the training set."""

    size = len(trainloader.dataset)
    num_batches = len(trainloader)
    train_acc, train_loss = 0, 0
    model.train()
    for batch, (X, y) in enumerate(trainloader):
        images = X.to(DEVICE)
        labels = y.to(DEVICE)
        if mixer:
            images, labels = mixer(images, labels)

        is_hot, hot_idx = is_hot_one(labels)
        if (batch % 100 == 0 or is_hot) and disp_log:
            if is_hot:
                print(f"hot-one detected at {hot_idx}: {labels[hot_idx] = }")
            print(
                f"\n{mixer = }\n{images.shape = }\n{images.dtype = }\n{images.ndim = }\n{labels.dtype = }\n{labels.shape = }\n{labels.ndim = }\n{labels.squeeze().shape = }\n"
            )
            print("row sums:", labels.sum(dim=1))
            print("min/max:", labels.min().item(), labels.max().item())
            print("")

        predictions = model(images)
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
        # loss = loss_func(model(images), labels)
        loss.backward()
        optimizer.step()
        ########

        if batch % 100 == 0 and disp_log:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_acc = train_acc / size
    train_loss = train_loss / num_batches

    return train_acc, train_loss


def test(model: NET, testloader: DataLoader, loss_func):
    """Validate the model on the test set."""
    size = len(testloader.dataset)
    num_batches = len(testloader)
    model.eval()
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            test_loss += loss_func(predictions, labels).item()
            test_acc += (predictions.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    test_acc /= size

    return test_acc, test_loss


def is_hot_one(labels: torch.Tensor) -> tuple[bool, int]:
    for idx, val in enumerate(labels):
        if 1.0 in val:
            return True, idx

    return False, 0
