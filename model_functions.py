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
    valid_labels: list,
    optimizer,
    loss_func,
    global_params: list,
    mixer=None,
    use_masking: bool = False,
    disp_log: bool = True,
    max_grad_norm: float = 1.0,
    mu: float = 0.0,
):
    """Train the model on the training set."""

    size = len(trainloader.dataset)
    num_batches = len(trainloader)
    train_acc, train_loss = 0, 0
    model.train()

    disp_window = []
    disp_window_size = 20

    # map global to local index (used only if masking)
    class_to_local = {c: i for i, c in enumerate(valid_labels)}

    for batch, (X, y) in enumerate(trainloader):

        images = X.to(DEVICE)
        labels_hard = y.to(DEVICE)

        # Apply mixer if exists
        if mixer:
            images, labels = mixer(images, labels_hard)  # soft labels
        else:
            labels = labels_hard  # hard labels

        predictions = model(images)  # [B, num_global_classes]

        # ==================================================
        # SWITCH: MASKED vs NORMAL TRAINING
        # ==================================================

        if use_masking:
            # ---- MASKED TRAINING ----
            logits = predictions[:, valid_labels]  # [B, num_valid]

            if labels.ndim > 1:
                # Soft labels (MixUp / CutMix)

                labels_local = labels[:, valid_labels]

                # renormalize probabilities
                labels_local = labels_local / labels_local.sum(
                    dim=1, keepdim=True
                ).clamp(min=1e-12)

                loss = loss_func(logits, labels_local)

                target_labels = labels_local.argmax(1)

            else:
                # Hard labels to remap to local indices
                targets = torch.tensor(
                    [class_to_local[int(lbl)] for lbl in labels], device=DEVICE
                )

                loss = loss_func(logits, targets)
                target_labels = targets

        else:
            logits = predictions
            loss = loss_func(logits, labels)

            # convert soft to hard for accuracy
            if labels.ndim > 1:
                target_labels = labels.argmax(1)
            else:
                target_labels = labels

        # ==================================================

        # FedProx term
        if mu > 0:
            prox_term = 0.0
            for p, gp in zip(model.parameters(), global_params):
                prox_term += (p - gp).pow(2).sum()
            loss += (mu / 2) * prox_term

        # predictions (always global)
        pred_labels = predictions.argmax(1)

        # when masking, target_labels are LOCAL → must convert back to global
        if use_masking:
            # map local index → global label
            local_to_class = {i: c for i, c in enumerate(valid_labels)}
            target_labels_global = torch.tensor(
                [local_to_class[int(t)] for t in target_labels], device=DEVICE
            )
        else:
            target_labels_global = target_labels

        # wccuracy
        train_acc += (
            (pred_labels == target_labels_global).type(torch.float).sum().item()
        )

        train_loss += loss.item()

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # logging window
        disp_window.append(loss.item())
        # if len(disp_window) >= disp_window_size:
        #     disp_window.pop(0)

        if batch % 100 == 0 and disp_log:
            # avg_loss = sum(disp_window) / len(disp_window)
            current = (batch + 1) * len(images)
            print(f"current loss: {loss.item():>7.6f}  [{current:>5d}/{size:>5d}]")
            # print(f"average loss: {avg_loss:>7.6f}  [{current:>5d}/{size:>5d}]")

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
        batch_count = 0
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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


def get_true_and_pred_values(
    testloader,
    model,
):
    actual_values = []
    pred_values = []
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            actual_values.extend(labels.cpu().tolist())
            pred_values.extend(predictions.cpu().tolist())

    return actual_values, pred_values


def acc_per_class(
    actual_values: list[int],
    pred_values: list[int],
    out_features: int,
    labels_map: dict[int, str],
) -> dict:
    global_labels_map = {
        i: c for i, c in enumerate(["Unknown-Class" for _ in range(out_features)])
    }
    global_labels_map.update(labels_map)

    classes = list(global_labels_map.values())
    correct_pred = {classname: 0.0 for classname in classes}
    total_pred = {classname: 0.0 for classname in classes}
    for label, prediction in zip(actual_values, pred_values):
        if label == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accuracy = 0.0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]

        correct_pred[classname] = accuracy

    return correct_pred


def display_acc_logs(eval_metrics: dict):
    for classname, accuracy in eval_metrics.items():
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f}%")


def eval_per_class(testloader, model, out_features: int, labels_map: dict):
    true_values, pred_values = get_true_and_pred_values(testloader, model)
    eval_results = acc_per_class(true_values, pred_values, out_features, labels_map)
    display_acc_logs(eval_results)


class EarlyStop:

    def __init__(self, tolerance: int = 1, min_delta: float = 0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.values = []
        self.current_delta = 0

    def delta(self, training_loss, validation_loss) -> float:
        self.current_delta = (validation_loss - training_loss) / (validation_loss)
        return self.current_delta

    def record(self, current_epoch, training_loss, validation_loss):
        if len(self.values) >= 2:
            self.values.pop(0)
        self.values.append((current_epoch, self.delta(training_loss, validation_loss)))

    def delta_slope(self):
        if len(self.values) == 2:
            e1, d1 = self.values[0]
            e2, d2 = self.values[1]
            return (d2 - d1) / (e2 - e1)
        else:
            return 0

    def early_stopper(self) -> bool:
        if self.delta_slope() > 0:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter >= self.tolerance:
            return True
        else:
            return False
