import torch as T
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

from models.kan_layers import KanLayer  # type: ignore

EXAMPLE_NAME = Path(__file__).stem

NUM_EPOCHS = 1_000
GRID_SIZE = 10
LR = 1e-2
NUM_POINTS = 1000
DOMAIN = [-1, 1]

F = lambda x: x**2 - T.sin((2 * x) ** 3) + T.cos((3 * x) ** 2)


def main() -> None:
    layers = [(layer, layer.value(1, 1, GRID_SIZE)) for layer in KanLayer]
    optimizers = [optim.Adam(l[1].parameters(), LR) for l in layers]
    criterion = nn.MSELoss()

    X = T.linspace(DOMAIN[0], DOMAIN[1], NUM_POINTS).unsqueeze(1)
    Y = F(X)

    writer = SummaryWriter("./runs/")

    for x, y in zip(X * (NUM_POINTS - 1), Y):
        writer.add_scalars(
            f"{EXAMPLE_NAME}/starting_graph",
            {"ground_truth": y.item()},
            x.item(),
        )
    with T.no_grad():
        for layer_type, layer in layers:
            Y_hat = layer(X)
            for x, y in zip(X * (NUM_POINTS - 1), Y_hat):
                writer.add_scalars(
                    f"{EXAMPLE_NAME}/starting_graph",
                    {layer_type.name: y.item()},
                    x.item(),
                )
                
    for epoch in tqdm(range(NUM_EPOCHS)):
        for (layer_type, layer), optimizer in zip(layers, optimizers):
            Y_hat = layer(X)
            loss = criterion(Y, Y_hat)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalars(
                f"{EXAMPLE_NAME}/training_loss",
                {layer_type.name: loss.detach()},
                epoch,
            )
    
    for x, y in zip(X * (NUM_POINTS - 1), Y):
        writer.add_scalars(
            f"{EXAMPLE_NAME}/ending_graph",
            {"ground_truth": y.item()},
            x.item(),
        )
    with T.no_grad():
        for layer_type, layer in layers:
            Y_hat = layer(X)
            for x, y in zip(X * (NUM_POINTS - 1), Y_hat):
                writer.add_scalars(
                    f"{EXAMPLE_NAME}/ending_graph",
                    {layer_type.name: y.item()},
                    x.item(),
                )


if __name__ == "__main__":
    main()
