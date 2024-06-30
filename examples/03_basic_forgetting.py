from pathlib import Path

import torch as T
from kan import KAN  # type: ignore
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Self

N = 10
STD = 0.2
NUM_SAMPLES = 1_000

LR = 1e-2
NUM_EPOCHS = 200
NUM_PARTITIONED_EPOCHS = 50


EXAMPLE_NAME = Path(__file__).stem


class MLP(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.ftc1 = nn.Linear(1, 64)
        self.ftc2 = nn.Linear(64, 64)
        self.ftc3 = nn.Linear(64, 1)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = F.leaky_relu(self.ftc1(x))
        x = F.leaky_relu(self.ftc2(x))
        x = self.ftc3(x)

        return x


def guassian(x: T.Tensor, mean: float, std: float) -> T.Tensor:
    return T.exp(-((x - mean) ** 2) / (2 * std**2))


def create_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(0, N, NUM_SAMPLES, device=device).unsqueeze(1)
    y = T.zeros_like(x)
    for i in range(N):
        y += guassian(x, i + 0.5, STD)

    return x, y


def create_partitioned_dataset(
    device: T.device,
) -> tuple[list[T.Tensor], list[T.Tensor]]:
    X, Y = create_dataset(device)
    partitioned_X = T.chunk(X, N)
    partitioned_Y = T.chunk(Y, N)

    return partitioned_X, partitioned_Y


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    writer = SummaryWriter("./runs/")

    models = [
        ("MLP",  MLP().to(device)),
        ("KAN", KAN([1, 32, 1]).to(device)),
    ]
    optimizers = [optim.Adam(model.parameters(), LR) for _, model in models]

    X, Y = create_dataset(device)
    partitioned_X, partitioned_Y = create_partitioned_dataset(device)

    # log model parameter counts
    writer.add_scalars(
        f"{EXAMPLE_NAME}/parameter_counts",
        {
            model_type: sum(p.numel() for p in model.parameters())
            for model_type, model in models
        },
        0,
    )

    for epoch in tqdm(range(NUM_EPOCHS)):
        for model_name, model in models:
            pass


if __name__ == "__main__":
    main()
