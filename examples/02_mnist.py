from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms  # type: ignore
from tqdm import tqdm
from typing_extensions import Self

from models import KanLayer, KanModel

MNIST_TRAIN_PATH = "./datasets/MNIST/mnist_train.csv"
MNIST_TEST_PATH = "./datasets/MNIST/mnist_test.csv"
IMG_SIZE = 28

GRID_SIZE = 15
NUM_EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
TESTING_BATCH_SIZE = 512

ROLLING_LOSS_N = 50
LOGGING_FREQ = 50

EXAMPLE_NAME = Path(__file__).stem


class DatasetType(Enum):
    training = 0
    testing = 1


class MnistDataset(Dataset):
    def __init__(
        self: Self,
        dataset_type: DatasetType,
        transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        self.dataset_type = dataset_type
        self.dataset_path = (
            MNIST_TRAIN_PATH
            if dataset_type == DatasetType.training
            else MNIST_TEST_PATH
        )
        self.data = pd.read_csv(self.dataset_path).to_numpy()

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self: Self) -> int:
        return self.data.shape[0]

    def __getitem__(self: Self, idx: int) -> tuple[Any, Any]:
        img = self.data[idx, 1:]
        label = self.data[idx, 0]

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label


class MLP(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.ftc1 = nn.Linear(IMG_SIZE**2, 128)
        self.ftc2 = nn.Linear(128, 128)
        self.ftc3 = nn.Linear(128, 10)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = F.leaky_relu(self.ftc1(x))
        x = F.leaky_relu(self.ftc2(x))
        x = self.ftc3(x)
        return x


@T.no_grad()
def calculate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    num_correct = 0.0
    num_total = 0

    for image_batch, label_batch in dataloader:
        Y_hat = model(image_batch)

        num_correct += T.sum(T.argmax(Y_hat, dim=-1) == label_batch).item()
        num_total += image_batch.shape[0]

    return num_correct / num_total


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    image_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255),
            transforms.Lambda(lambda x: T.tensor(x, dtype=T.float32, device=device)),
        ]
    )
    target_transforms = transforms.Compose(
        [transforms.Lambda(lambda x: T.tensor(x, dtype=T.long, device=device))]
    )

    training_dataset = MnistDataset(
        DatasetType.training,
        transforms=image_transforms,
        target_transforms=target_transforms,
    )
    testing_dataset = MnistDataset(
        DatasetType.testing,
        transforms=image_transforms,
        target_transforms=target_transforms,
    )
    training_dataloader = DataLoader(training_dataset, BATCH_SIZE)
    testing_dataloader = DataLoader(testing_dataset, TESTING_BATCH_SIZE)

    models = [
        (layer.name, KanModel([IMG_SIZE**2, 10], layer, GRID_SIZE).to(device))
        for layer in KanLayer
    ] + [("MLP", MLP().to(device))]
    optimizers = [optim.Adam(m.parameters(), LR) for _, m in models]
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("./runs/")

    # log model parameter counts
    writer.add_scalars(
        f"{EXAMPLE_NAME}/parameter_counts",
        {
            model_type: sum(p.numel() for p in model.parameters())
            for model_type, model in models
        },
        0,
    )

    losses = [0.0 for _ in range(len(models))]
    num_corrects = [0.0 for _ in range(len(models))]
    num_total = 0
    iteration = 0
    for epoch in range(NUM_EPOCHS):
        for img_batch, label_batch in tqdm(
            training_dataloader, desc=f"Epoch {epoch + 1}"
        ):
            for i, ((_, model), optimizer) in enumerate(
                zip(models, optimizers)
            ):
                Y_hat = model(img_batch)
                loss = criterion(Y_hat, label_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                num_corrects[i] += T.sum(T.argmax(Y_hat, dim=1) == label_batch).item()

                losses[i] += (loss.item() - losses[i]) / ROLLING_LOSS_N

            num_total += img_batch.shape[0]

            if iteration % LOGGING_FREQ == 0:
                writer.add_scalars(
                    f"{EXAMPLE_NAME}/training_loss",
                    {model_type: loss for (model_type, _), loss in zip(models, losses)},
                    iteration,
                )

                writer.add_scalars(
                    f"{EXAMPLE_NAME}/testing_accuracy",
                    {
                        model_type: calculate_accuracy(model, testing_dataloader)
                        for model_type, model in models
                    },
                    iteration,
                )

                writer.add_scalars(
                    f"{EXAMPLE_NAME}/training_accuracy",
                    {
                        model_type: nc / num_total
                        for (model_type, _), nc in zip(models, num_corrects)
                    },
                    iteration,
                )

                num_corrects = [0.0 for _ in range(len(models))]
                num_total = 0

            iteration += 1


if __name__ == "__main__":
    main()
