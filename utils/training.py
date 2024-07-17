from typing import Callable, Type

import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def RMSE_loss(input: T.Tensor, target: T.Tensor) -> T.Tensor:
    return T.sqrt(F.mse_loss(input, target))


def train_model(
    model: nn.Module,
    datasets: dict[str, Dataset],
    optimizer: Type[optim.Optimizer] = optim.SGD,
    loss_fn: Callable = nn.MSELoss(),
    epochs: int = 500,
    lr: float = 1e-2,
    batch_size: int = 8,
    eval_loss_fn: Callable = RMSE_loss,
    eval_batch_size: int = 32,
    logging_freq: int = 100,
) -> dict[str, list[float]]:
    """
    Trains a model according to parameters and datasets
    while saving and returning metrics

    Parameters
    ----------
    model : nn.Module
        Pytorch model to train

    datasets : dict[str, Dataset]
        Dataset to train / evaluate on, must contain key "train"

    optimizer : Type[optim.Optimizer], optional
        Pytorch optimize to use, by default optim.SGD

    loss_fn : Callable, optional
        Loss function for training, by default nn.MSELoss()

    epochs : int, optional
        Number of epochs to train for, by default 500

    lr : float, optional
        Learning rate passed to optimizer as kwarg, by default 1e-2

    batch_size : int, optional
        Batch size for training dataloader, by default 8

    eval_loss_fn : Callable, optional
        Function to evaluate non-train datasets on, by default RMSE_loss

    eval_batch_size : int, optional
        Batch size for evaluation dataloaders, by default 32

    logging_freq : int, optional
        Frequency to save train loss and evaluation losses, by default 100

    Returns
    -------
    dict[str, list[float]]
        Loss metrics for each dataset in datasets 
    """

    model_optimizer = optimizer(model.parameters(), lr=lr)  # type: ignore

    # initialize train / eval dataloaders and results dict
    train_dataset = datasets["train"]
    train_dataloader = DataLoader(train_dataset, batch_size)
    eval_dataloaders = {
        n: DataLoader(d, eval_batch_size) for n, d in datasets.items() if n != "train"
    }
    results: dict[str, list[float]] = {d: [] for d in datasets.keys()}

    rolling_loss = 0
    iteration = 0

    for _ in range(epochs):
        for X_batch, Y_batch in train_dataloader:
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)
            loss.backward()

            with T.no_grad():
                eval_loss = eval_loss_fn(Y_batch, Y_pred).item()

            # update rolling loss
            if iteration == 0:
                rolling_loss = eval_loss
            else:
                n = min(iteration, logging_freq)
                rolling_loss = (rolling_loss * (n - 1) + eval_loss) / n

            model_optimizer.step()
            model_optimizer.zero_grad()

            iteration += 1

            if iteration % logging_freq == 0:
                results["train"].append(rolling_loss)

                # calculate average loss of each eval_dataloader
                with T.no_grad():
                    for name, dataloader in eval_dataloaders.items():
                        losses = []
                        for X_batch, Y_batch in dataloader:
                            Y_pred = model(X_batch)
                            loss = eval_loss_fn(Y_batch, Y_pred)
                            losses.append(loss.item())

                        results[name].append(sum(losses) / len(losses))

    return results
