from dataclasses import dataclass
from typing import Any, Callable, Type

import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import Self


def calculate_accuracy(input: T.Tensor, target: T.Tensor) -> T.Tensor:
    """
    Calculates accuracy on two tensors by argmaxing last dim first

    Parameters
    ----------
    input : T.Tensor
        Input tensor

    target : T.Tensor
        Target tensor to compare to

    Returns
    -------
    T.Tensor
        Dim 0 tensor representing accuracy from 0 => 100
    """
    
    arged_input = T.argmax(input, dim=-1)
    arged_target = T.argmax(target, dim=-1)

    accuracy = T.mean((arged_input == arged_target).to(T.float32))

    return 100 * accuracy


def RMSE_loss(input: T.Tensor, target: T.Tensor, **mse_kwargs: Any) -> T.Tensor:
    """
    Calculates Root Mean Squared Error

    Parameters
    ----------
    input : T.Tensor
        Input tensor

    target : T.Tensor
        Target tensor to compare to

    mse_kwargs : dict[str, Any]
        Kwargs to pass to F.mse_loss

    Returns
    -------
    T.Tensor
        Dim 0 tensor representing root mean squared errors
    """
    return T.sqrt(F.mse_loss(input, target, **mse_kwargs))


@dataclass
class TrainModelArguments:
    model: nn.Module | None = None
    datasets: dict[str, Dataset] | None = None
    optimizer: Type[optim.Optimizer] | None = None
    loss_fn: Callable | None = None
    num_epochs: int | None = None
    lr: float | None = None
    batch_size: int | None = None
    eval_fns: dict[str, Callable[[T.Tensor, T.Tensor], T.Tensor]] | None = None
    eval_batch_size: int | None = None
    logging_freq: int | None = None
    pbar_description: str | None = None

    def to_dict(self: Self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def train_model(
    model: nn.Module,
    datasets: dict[str, Dataset],
    optimizer: Type[optim.Optimizer] = optim.SGD,
    loss_fn: Callable = nn.MSELoss(),
    num_epochs: int = 500,
    lr: float = 1e-2,
    batch_size: int = 8,
    eval_fns: dict[str, Callable[[T.Tensor, T.Tensor], T.Tensor]] = {"loss": RMSE_loss},
    eval_batch_size: int = 32,
    logging_freq: int = 100,
    pbar_description: str = "",
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

    num_epochs : int, optional
        Number of epochs to train for, by default 500

    lr : float, optional
        Learning rate passed to optimizer as kwarg, by default 1e-2

    batch_size : int, optional
        Batch size for training dataloader, by default 8

    eval_fns : dict[str, Callable[[T.Tensor, T.Tensor], T.Tensor]], optional
        Functions to evaluate datasets on, by default {"loss": MSE_loss}

    eval_batch_size : int, optional
        Batch size for evaluation dataloaders, by default 32

    logging_freq : int, optional
        Frequency to save train loss and evaluation losses, by default 100

    pbar_description : str, optional
        Description to use on progress bar, by default ""

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
    results: dict[str, list[float]] = {
        f"{d}_{e}": [] for d in datasets.keys() for e in eval_fns
    }

    train_metrics = {k: 0.0 for k in eval_fns}
    iteration = 0

    pbar: tqdm = tqdm(total=num_epochs * len(train_dataloader))
    pbar.set_description(f"{pbar_description}: {' | '.join(f'{k}: #####' for k in results)}")  # type: ignore

    for _ in range(num_epochs):
        for X_batch, Y_batch in train_dataloader:
            Y_pred = model(X_batch)

            loss = loss_fn(Y_pred, Y_batch)
            loss.backward()

            model_optimizer.step()
            model_optimizer.zero_grad()

            with T.no_grad():
                # training metrics are evaluated as rolling
                for metric, eval_fn in eval_fns.items():
                    metric_val = eval_fn(Y_pred, Y_batch).item()

                    if iteration == 0:
                        train_metrics[metric] = metric_val
                    else:
                        n = min(iteration, logging_freq)
                        train_metrics[metric] = (
                            train_metrics[metric] * (n - 1) + metric_val
                        ) / n

            iteration += 1

            pbar.update()

            if iteration % logging_freq == 0:
                for metric, metric_value in train_metrics.items():
                    results[f"train_{metric}"].append(metric_value)

                # calculate average loss of each eval_dataloader
                with T.no_grad():
                    for name, dataloader in eval_dataloaders.items():
                        for metric, eval_fn in eval_fns.items():
                            metric_values = []
                            for X_batch, Y_batch in dataloader:
                                Y_pred = model(X_batch)
                                metric_val = eval_fn(Y_pred, Y_batch).item()
                                metric_values.append(metric_val)

                            results[f"{name}_{metric}"].append(
                                sum(metric_values) / len(metric_values)
                            )

                # add losses to progress bar
                str_losses = " | ".join(f"{k}: {v[-1]:.2f}" for k, v in results.items())
                pbar.set_description(f"{pbar_description}: {str_losses}")  # type: ignore

    pbar.close()

    return results
