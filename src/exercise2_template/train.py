from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import typer
from torch import nn


import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

DATA_PATH = "data/raw"


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set
# from model import MyAwesomeModel

# from data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# app = typer.Typer()

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

# @app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")





def main():
    train() 







# from model import MyAwesomeModel

# from data import corrupt_mnist
# app = typer.Typer()
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# @app.command()
# def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
#     """Train a model on MNIST."""
#     print("Training day and night")
#     print(f"{lr=}, {batch_size=}, {epochs=}")

#     model = MyAwesomeModel().to(DEVICE)
#     train_set, _ = corrupt_mnist()

#     train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     statistics = {"train_loss": [], "train_accuracy": []}
#     for epoch in range(epochs):
#         model.train()
#         for i, (img, target) in enumerate(train_dataloader):
#             img, target = img.to(DEVICE), target.to(DEVICE)
#             optimizer.zero_grad()
#             y_pred = model(img)
#             loss = loss_fn(y_pred, target)
#             loss.backward()
#             optimizer.step()
#             statistics["train_loss"].append(loss.item())

#             accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
#             statistics["train_accuracy"].append(accuracy)

#             if i % 100 == 0:
#                 print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

#     print("Training complete")
#     torch.save(model.state_dict(), "models/model.pth")
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#     axs[0].plot(statistics["train_loss"])
#     axs[0].set_title("Train loss")
#     axs[1].plot(statistics["train_accuracy"])
#     axs[1].set_title("Train accuracy")
#     fig.savefig("reports/figures/training_statistics.png")


# @app.command()
# def evaluate(model_checkpoint: str) -> None:
#     """Evaluate a trained model."""
#     print("Evaluating like my life depended on it")
#     print(model_checkpoint)

#     model = MyAwesomeModel().to(DEVICE)
#     model.load_state_dict(torch.load(model_checkpoint))

#     _, test_set = corrupt_mnist()
#     test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

#     model.eval()
#     correct, total = 0, 0
#     for img, target in test_dataloader:
#         img, target = img.to(DEVICE), target.to(DEVICE)
#         y_pred = model(img)
#         correct += (y_pred.argmax(dim=1) == target).float().sum().item()
#         total += target.size(0)
#     print(f"Test accuracy: {correct / total}")


# if __name__ == "__main__":
#     app()
#     # train()