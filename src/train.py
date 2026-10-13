"""Build a model and train it. YOU write the bodies of these functions.

The model is ResNet18, a convolutional network that was already trained on
ImageNet, 1.2 million photographs of 1,000 kinds of thing. You keep everything
it learned there and replace only its last layer, so that it answers with your
categories instead of those 1,000. Then you train it further on your images.

Run `pytest tests/test_train.py` after you fill them in. The first run
downloads the ImageNet weights, about 45 MB.
"""

import torch
import torch.nn as nn


def build_model(num_classes, pretrained=True, freeze=False):
    """Return a ResNet18 whose last layer has `num_classes` outputs.

    Arguments
        num_classes  int, how many categories you have
        pretrained   True: start from the ImageNet weights,
                     torchvision.models.ResNet18_Weights.IMAGENET1K_V1.
                     False: start from random numbers, as if ImageNet had
                     never happened. You need this for the comparison in
                     Problem 3.
        freeze       True: only the new last layer is trained. Every other
                     parameter keeps the value it came with, which you do by
                     setting requires_grad to False on it.
                     False: every parameter is trained.

    Returns the model from torchvision.models.resnet18, with its `fc`
    attribute replaced by a new nn.Linear that has 512 inputs and num_classes
    outputs.

    Do not put a softmax at the end. The loss function adds it for you, and the
    web page adds it for you.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(model, X_train, y_train, X_test, y_test,
          epochs=10, lr=1e-4, batch_size=32):
    """Train the model and report what happened at every epoch.

    Arguments
        model        what build_model returned, or any other torch model
        X_train      np.float32 (N, ...)      y_train  np.int64 (N,)
        X_test       np.float32 (M, ...)      y_test   np.int64 (M,)
        epochs       int, how many times to go through the training set
        lr           float, the learning rate
        batch_size   int, how many rows per step

    Returns a dict named history with four keys. Each value is a list of
    `epochs` numbers, one per epoch:
        "train_loss"  the mean loss over the batches of that epoch
        "train_acc"   the share of training rows the model got right in those
                      batches, between 0.0 and 1.0
        "test_loss"   the mean loss over the test set, measured after the epoch
        "test_acc"    the share of the test set it gets right, measured after
                      the epoch, between 0.0 and 1.0

    What to do
        Use a GPU when there is one:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Move the model there once, and each batch there as you use it. Keep
        X_train and X_test themselves in ordinary memory: they may not fit on
        the GPU.
        Shuffle the training rows again at the start of every epoch.
        Use nn.CrossEntropyLoss, which expects raw outputs, and torch.optim.Adam
        over the parameters that have requires_grad set.
        Call model.train() before the training batches and model.eval() before
        measuring the test set. ResNet contains BatchNorm layers, which behave
        differently in the two modes.
        Do not compute gradients while measuring the test set, and never let
        the test rows change the parameters.
        Print one line per epoch, so that you can watch it while it runs.

    The model is trained in place. When this function returns, `model` is the
    trained one, and it is left on the device it was trained on.
    """
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training and test sets must both contain rows")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Each set must have one label per row")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    optimiser = torch.optim.Adam(parameters, lr=lr)
    criterion = nn.CrossEntropyLoss()

    def loader(X, y, shuffle):
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32, device="cpu"),
            torch.as_tensor(y, dtype=torch.long, device="cpu"),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    training = loader(X_train, y_train, True)
    testing = loader(X_test, y_test, False)
    history = {key: [] for key in ("train_loss", "train_acc", "test_loss", "test_acc")}

    for epoch in range(epochs):
        model.train()
        # A frozen BatchNorm layer must also keep its running statistics fixed.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        train_loss, train_correct = 0.0, 0
        for rows, labels in training:
            rows, labels = rows.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(rows)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()

        model.eval()
        test_loss, test_correct = 0.0, 0
        with torch.no_grad():
            for rows, labels in testing:
                rows, labels = rows.to(device), labels.to(device)
                logits = model(rows)
                test_loss += criterion(logits, labels).item() * len(labels)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()

        history["train_loss"].append(train_loss / len(training.dataset))
        history["train_acc"].append(train_correct / len(training.dataset))
        history["test_loss"].append(test_loss / len(testing.dataset))
        history["test_acc"].append(test_correct / len(testing.dataset))
        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train loss {history['train_loss'][-1]:.4f}, "
            f"train accuracy {history['train_acc'][-1]:.4f}, "
            f"test loss {history['test_loss'][-1]:.4f}, "
            f"test accuracy {history['test_acc'][-1]:.4f}",
            flush=True,
        )
    return history


def count_trainable(model):
    """How many parameters training is allowed to change. Written for you."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, path):
    """Save a trained model. Written for you."""
    torch.save(model, path)


def load(path):
    """Load a model saved by `save`, onto the CPU. Written for you."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model
