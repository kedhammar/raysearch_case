import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from skimage.filters import gaussian
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.softmax(self.fc2(self.relu(self.fc1(x))))


def prepare_mnist_datasets(root, train_path, test_path):
    train_dataset = datasets.MNIST(
        root=root, train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, transform=transforms.ToTensor(), download=True
    )

    train_tensors = torch.stack([img[0] for img in train_dataset])
    train_labels = torch.tensor([label for _, label in train_dataset])
    torch.save((train_tensors, train_labels), train_path)

    test_tensors = torch.stack([img[0] for img in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    torch.save((test_tensors, test_labels), test_path)


def load_dataset(data_path, max_samples=None):
    tensors, labels = torch.load(os.path.join(data_path))
    if max_samples:
        tensors = tensors[:max_samples]
        labels = labels[:max_samples]
    return TensorDataset(tensors, labels)


def save_dataset(path, dataset):
    images, labels = dataset.tensors
    torch.save((images, labels), os.path.join(path))


def create_blurred_dataset(input_path, output_path, sigma=2):
    dataset = load_dataset(input_path)
    images, labels = dataset.tensors
    blurred_images = np.empty_like(images)

    for i in range(images.shape[0]):
        blurred_images[i, 0] = gaussian(images[i, 0], sigma=sigma, mode="reflect")

    blurred_dataset = TensorDataset(torch.Tensor(blurred_images), labels)
    save_dataset(output_path, blurred_dataset)


def train_model(config):
    full_dataset = load_dataset(config["data_path"], config["max_samples"])
    train_size = int(config["data_split_ratio"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = SimpleNN(config["hidden_size"])
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["max_epochs"]):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch + 1}/{config['max_epochs']}], "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}"
        )

    torch.save(
        {"model_state_dict": model.state_dict(), "hidden_size": config["hidden_size"]},
        config["output_path"],
    )


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return {"loss": running_loss / total, "accuracy": correct / total}


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {"loss": running_loss / total, "accuracy": correct / total}


def evaluate_model(model_path, data_path):
    checkpoint = torch.load(model_path)
    model = SimpleNN(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = load_dataset(data_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    metrics = evaluate(model, test_loader, nn.NLLLoss())

    print(f"Test Loss: {metrics['loss']:.4f}, Test Accuracy: {metrics['accuracy']:.4f}")


def make_configs(mnist_train, mnist_train_blurred):
    """Make config variants. Input a train data set and a blurred train data set."""
    # Default config
    default_config = {
        "batch_size": 64,
        "hidden_size": 128,
        "learning_rate": 0.01,
        "data_path": mnist_train,
        "data_split_ratio": 0.8,
        "max_epochs": 10,
        "max_samples": None,
    }

    # Config variants
    configs = {
        "default": default_config,
        "hidden2": default_config.copy(),
        "samples100": default_config.copy(),
        "blurred": default_config.copy(),
    }
    configs["hidden2"]["hidden_size"] = 2
    configs["samples100"]["max_samples"] = 100
    configs["blurred"]["data_path"] = mnist_train_blurred
    for config_name, config in configs.items():
        config["output_path"] = root / f"model_{config_name}.pth"

    return configs


if __name__ == "__main__":
    root = Path("./root")
    mnist_train = root / "mnist_train.pt"
    mnist_test = root / "mnist_test.pt"
    prepare_mnist_datasets(root, mnist_train, mnist_test)

    # Create blurred datasets
    mnist_train_blurred = root / "mnist_train_blurred.pt"
    create_blurred_dataset(mnist_train, mnist_train_blurred)

    mnist_test_blurred = root / "mnist_test_blurred.pt"
    create_blurred_dataset(mnist_test, mnist_test_blurred)

    # Make configs
    configs = make_configs(mnist_train, mnist_train_blurred)

    # Start training and evaluation
    for config_name, config in configs.items():
        train_model(config)
        evaluate_model(config["output_path"], mnist_test)
        evaluate_model(config["output_path"], mnist_test_blurred)
