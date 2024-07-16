from typing import Callable, Type

import torch as T
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import math


def train_model(
    model: nn.Module,
    datasets: dict[str, Dataset],
    optimizer: Type[optim.Optimizer] = optim.Adam,
    loss_fn: Callable = nn.MSELoss(),
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 8,
    eval_loss_fn: Callable = nn.MSELoss(),
    eval_batch_size: int = 32,
    logging_freq: int = 100,
) -> dict[str, list[float]]:
    model_optimizer = optimizer(model.parameters(), lr=lr)  # type: ignore

    train_dataset = datasets["train"]
    train_dataloader = DataLoader(train_dataset, batch_size)
    eval_dataloaders = {
        n: DataLoader(d, eval_batch_size) for n, d in datasets.items() if n != "train"
    }
    results: dict[str, list[float]] = {d: [] for d in datasets.keys()}

    rolling_loss = 0
    iteration = 0

    for epoch in range(epochs):
        for X_batch, Y_batch in train_dataloader:
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)
            loss.backward()

            if iteration == 0:
                rolling_loss = loss.item()
            else:
                n = min(iteration, logging_freq)
                rolling_loss = (rolling_loss * (n - 1) + loss.item()) / n

            model_optimizer.step()
            model_optimizer.zero_grad()

            iteration += 1

            if iteration % logging_freq == 0:
                results["train"].append(math.sqrt(rolling_loss))

                with T.no_grad():
                    for name, dataloader in eval_dataloaders.items():
                        losses = []
                        for X_batch, Y_batch in dataloader:
                            Y_pred = model(X_batch)
                            loss = eval_loss_fn(Y_batch, Y_pred)
                            losses.append(loss.item())

                        results[name].append(math.sqrt(sum(losses) / len(losses)))

    return results
