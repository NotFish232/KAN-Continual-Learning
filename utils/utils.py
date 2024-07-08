import math
import torch as T
from torch import nn, optim
from typing import Callable


def suggest_MLP_architecture(
    num_inputs: int, num_outputs: int, num_layers: int, num_params: int
) -> list[int]:
    """
    creates an MLP architecture satisfying the constraints and having roughly n parameters
    """
    hidden_dim = int(math.sqrt(num_params / max(num_layers - 2, 1)) - 1)

    architecture = [num_inputs, *([hidden_dim] * (num_layers - 1)), num_outputs]

    return architecture


def suggest_KAN_architecture(
    num_inputs: int,
    num_outputs: int,
    num_layers: int,
    num_params: int,
) -> tuple[list[int], int]:
    """
    creates an KAN architecture satisfying the constraints and having roughly n parameters
    """
    grid_size = num_params // max(num_layers - 2, 1) ** 2

    hidden_dim = int(math.sqrt(num_params / max(grid_size * (num_layers - 2), 1)) - 1)

    architecture = [num_inputs, *([hidden_dim] * (num_layers - 1)), num_outputs]

    return architecture, grid_size


def num_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    dataset: dict[str, T.Tensor],
    num_epochs: int,
    LR: float = 1e-2,
    loss_fn: Callable | None = None,
) -> None:
    """
    trains a model using SGD for num_epochs
    dataset should be formatted the same way it is for pykan
    """
    X = dataset["train_input"]
    Y = dataset["train_label"]

    optimizer = optim.Adam(model.parameters(), LR)
    criterion = loss_fn or nn.MSELoss()

    for epoch in range(num_epochs):
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
