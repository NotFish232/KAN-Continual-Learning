import math
from typing import Any, Callable

import torch as T
from kan import KAN
from plotly import graph_objects as go  # type: ignore
from torch import nn, optim
from torch.nn import functional as F


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
) -> dict[str, Any]:
    """
    trains a model using SGD for num_epochs
    dataset should be formatted the same way it is for pykan
    """
    X_train = dataset["train_input"]
    Y_train = dataset["train_label"]
    X_test = dataset["test_input"]
    Y_test = dataset["test_label"]

    optimizer = optim.Adam(model.parameters(), LR)
    criterion = loss_fn or nn.MSELoss()

    results: dict[str, Any] = {
        "train_loss": [],
        "test_loss": [],
    }

    for epoch in range(num_epochs):
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with T.no_grad():
            results["train_loss"].append(T.sqrt(F.mse_loss(model(X_train), Y_train)))
            results["test_loss"].append(T.sqrt(F.mse_loss(model(X_test), Y_test)))

    return results


def mse_reg_loss(
    kan: KAN,
    lamb: float = 0.0,
    lamb_l1: float = 1.0,
    lamb_entropy: float = 2.0,
    lamb_coef: float = 0.0,
    lamb_coefdiff: float = 0.0,
    small_mag_threshold: float = 1e-16,
    small_reg_factor: float = 1.0,
) -> Callable:
    def reg() -> T.Tensor:
        def nonlinear(
            x: T.Tensor,
            th: float = small_mag_threshold,
            factor: float = small_reg_factor,
        ) -> T.Tensor:
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = T.tensor(0.0)
        for i in range(len(kan.acts_scale)):
            vec = kan.acts_scale[i].reshape(-1)

            p = vec / T.sum(vec)
            l1 = T.sum(nonlinear(vec))
            entropy = -T.sum(p * T.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(kan.act_fun)):
            coeff_l1 = T.sum(T.mean(T.abs(kan.act_fun[i].coef), dim=1))
            coeff_diff_l1 = T.sum(T.mean(T.abs(T.diff(kan.act_fun[i].coef)), dim=1))
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    def _mse_reg_loss(Y_pred: T.Tensor, Y: T.Tensor) -> T.Tensor:
        return F.mse_loss(Y_pred, Y) + lamb * reg()

    return _mse_reg_loss


def plot_on_subplot(
    plot: go.Figure, pos: tuple[int, int], *subplots: go.Figure
) -> None:
    for subplot in subplots:
        for figure in subplot.data:
            plot.add_trace(figure, pos[0], pos[1])


def gaussian(x: T.Tensor, mean: float, std: float) -> T.Tensor:
    return T.exp(-((x - mean) ** 2) / (2 * std**2))
