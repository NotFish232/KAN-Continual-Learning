from pathlib import Path

import torch as T
from kan import KAN  # type: ignore
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Self


N = 5
STD = 0.2
NUM_POINTS = 1_000

LR = 1e-2
NUM_EPOCHS = 2_000
NUM_PARTITIONED_EPOCHS = 500


EXAMPLE_NAME = Path(__file__).stem


class MLP(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.ftc1 = nn.Linear(1, 32)
        self.ftc2 = nn.Linear(32, 32)
        self.ftc3 = nn.Linear(32, 1)

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = F.tanh(self.ftc1(x))
        x = F.tanh(self.ftc2(x))
        x = self.ftc3(x)

        return x


def guassian(x: T.Tensor, mean: float, std: float) -> T.Tensor:
    return T.exp(-((x - mean) ** 2) / (2 * std**2))


def create_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(0, N, NUM_POINTS, device=device).unsqueeze(1)
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


def reset_parameters(module: nn.Module) -> None:
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()  # type: ignore
        reset_parameters(layer)


def main() -> None:
    device = T.device("cpu")  # pykan broken on gpu :/

    writer = SummaryWriter("./runs/")

    models = [
        ("MLP", MLP().to(device)),
        ("KAN", KAN([1, 12, 12, 1]).to(device)),
    ]
    optimizers = [optim.Adam(model.parameters(), LR) for _, model in models]
    criterion = nn.MSELoss()

    X, Y = create_dataset(device)
    X_partitioned, Y_partitioned = create_partitioned_dataset(device)

    # log model parameter counts
    writer.add_scalars(
        f"{EXAMPLE_NAME}/parameter_counts",
        {
            model_type: sum(p.numel() for p in model.parameters())
            for model_type, model in models
        },
        0,
    )

    for _ in tqdm(range(NUM_EPOCHS)):
        for (_, model), optimizer in zip(models, optimizers):
            Y_pred = model(X)
            loss = criterion(Y, Y_pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    for x, y in zip(X * (NUM_POINTS // N - 1), Y):
        writer.add_scalars(
            f"{EXAMPLE_NAME}/normal_training", {"ground_truth": y.item()}, x.item()
        )
    with T.no_grad():
        for model_name, model in models:
            Y_pred = model(X)

            for x, y in zip(X * (NUM_POINTS // N - 1), Y_pred):
                writer.add_scalars(
                    f"{EXAMPLE_NAME}/normal_training", {model_name: y.item()}, x.item()
                )

    for i, (_, model) in enumerate(models):
        reset_parameters(model)
        optimizers[i] = optim.Adam(model.parameters(), LR)


    for n, (X_batch, Y_batch) in enumerate(zip(X_partitioned, Y_partitioned)):
        for _ in tqdm(range(NUM_PARTITIONED_EPOCHS), desc=f"Trial {n}"):
            for (_, model), optimizer in zip(models, optimizers):
                Y_pred = model(X_batch)
                loss = criterion(Y_batch, Y_pred)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

        X_indices = (X_batch * (NUM_POINTS // N - 1)).to(T.int32).squeeze()

        ground_truth = T.zeros_like(Y)
        ground_truth[X_indices] = Y_batch
        for x, y in zip(X * (NUM_POINTS // N - 1), ground_truth):
            writer.add_scalars(
                f"{EXAMPLE_NAME}/trial_{n}",
                {"ground_truth": y.item()},
                x.item(),
            )

        with T.no_grad():
            for model_name, model in models:
                Y_pred = model(X_batch)

                full_Y = T.zeros_like(Y)
                full_Y[X_indices] = Y_pred

                for x, y in zip(X * (NUM_POINTS // N - 1), full_Y):
                    writer.add_scalars(
                        f"{EXAMPLE_NAME}/trial_{n + 1}", {model_name: y.item()}, x.item()
                    )


if __name__ == "__main__":
    main()
