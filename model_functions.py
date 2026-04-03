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


# def train(
#     model: NET,
#     trainloader: DataLoader,
#     optimizer,
#     loss_func,
#     mixer=None,
#     disp_log: bool = True,
#     max_grad_norm: float = 1.0,
# ):
#     """Train the model on the training set."""

#     size = len(trainloader.dataset)
#     num_batches = len(trainloader)
#     train_acc, train_loss = 0, 0
#     model.train()

#     for batch, pair in enumerate(trainloader):
#         (X, y) = pair
#         images = X.to(DEVICE)
#         labels = y.to(DEVICE)
#         if mixer:
#             images, labels = mixer(images, labels)
#         predictions = model(images)

#         # if (batch % 100 == 0) and disp_log:
#         #     print(f"{labels.shape = }\n{predictions.shape = }")
#         #     print(f"{labels.dtype = }\n{predictions.dtype = }")
#         #     print(f"{images.dtype = }\n")
#         #     print("")

#         loss = loss_func(predictions, labels)

#         pred_labels = predictions.argmax(1)

#         if labels.ndim > 1:
#             target_labels = labels.argmax(1)
#         else:
#             target_labels = labels

#         train_acc += (
#             (pred_labels == target_labels).type(torch.float).sum().item()
#         )  # here all correct preds are summed
#         train_loss += loss.item()  # this returns the average loss over the batch

#         # Backpropagation
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()

#         # to prevent gradient explosion
#         if max_grad_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

#         optimizer.step()
#         ########

#         # average loss display
#         batch_size = labels.size(dim=0)


#         if batch % 100 == 0 and disp_log:
#             loss, current = loss.item(), (batch + 1) * len(images)
#             print(f"loss: {loss:>7.6f}  [{current:>5d}/{size:>5d}]")

#     train_acc = train_acc / size
#     train_loss = train_loss / num_batches

#     return train_acc, train_loss


def train(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: nn.Module,
    mixer=None,
    disp_log: bool = True,
    max_grad_norm: float = 1.0,
):
    model.train()
    size = len(trainloader.dataset)
    num_batches = len(trainloader)

    # Running sums for accurate epoch metrics
    total_correct = 0
    total_loss_sum = 0.0
    total_samples_processed = 0

    # For smoothed logging
    loss_window = []
    WINDOW_SIZE = 20

    for batch, (images, labels) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 🔍 Sanity check: ensure labels match expected class range
        # If your global model has 8 classes, labels should be in [0, 7]
        # Uncomment below to catch index mismatches early:
        # assert labels.min() >= 0 and labels.max() < 8, f"Label range error: {labels.min()}-{labels.max()}"

        if mixer is not None:
            images, labels = mixer(images, labels)

        predictions = model(images)
        loss = loss_func(predictions, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # Accuracy tracking (valid for both hard and soft labels)
        pred_labels = predictions.argmax(1)
        target_labels = labels.argmax(1) if labels.ndim > 1 else labels
        batch_correct = (pred_labels == target_labels).type(torch.float).sum().item()

        batch_size = images.size(0)
        total_correct += batch_correct
        total_loss_sum += (
            loss.item() * batch_size
        )  # Convert mean->sum for proper weighting
        total_samples_processed += batch_size

        # Smoothed logging
        loss_window.append(loss.item())
        if len(loss_window) > WINDOW_SIZE:
            loss_window.pop(0)

        if batch % 100 == 0 and disp_log:
            avg_loss = sum(loss_window) / len(loss_window)
            current = (batch + 1) * len(images)
            print(f"avg_loss: {avg_loss:>7.4f}  [{current:>5d}/{size:>5d}]")

    return total_correct / size, total_loss_sum / size


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

            if ignore_labels:
                mask = torch.isin(labels, torch.tensor(ignore_labels).to(DEVICE))
                # labels[mask] = -100
                if mask.all():
                    continue

            predictions = model(images)
            loss = loss_func(predictions, labels)
            test_loss += loss.item()
            batch_count += 1
            test_acc += (predictions.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= batch_count
    # test_loss /= num_batches
    test_acc /= size

    return test_acc, test_loss
