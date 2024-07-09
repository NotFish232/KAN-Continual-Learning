from pathlib import Path

import torch as T
from kan import KAN
from matplotlib import pyplot as plt

from utils import suggest_KAN_architecture, suggest_MLP_architecture, train_model
from utils.io import ExperimentWriter
from utils.models import MLP
from tqdm import tqdm

EXPERIMENT_NAME = Path(__file__).stem

NUM_PEAKS = 5
NUM_POINTS = 500
GAUSSIAN_STD = 0.2

NUM_EPOCHS = 200
NUM_PARAMETERS = 200

MLP_ARCHITECTURE = suggest_MLP_architecture(
    num_inputs=1,
    num_outputs=1,
    num_layers=3,
    num_params=NUM_PARAMETERS,
)
KAN_ARCHITECTURE, KAN_GRID_SIZE = suggest_KAN_architecture(
    num_inputs=1,
    num_outputs=1,
    num_layers=1,
    num_params=NUM_PARAMETERS,
)


def gaussian(x: T.Tensor, mean: float, std: float) -> T.Tensor:
    return T.exp(-((x - mean) ** 2) / (2 * std**2))


def create_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(0, NUM_PEAKS, NUM_POINTS, device=device).unsqueeze(1)
    y = T.zeros_like(x)
    for i in range(NUM_PEAKS):
        y += gaussian(x, i + 0.5, GAUSSIAN_STD)

    return x, y


def create_partitioned_dataset(
    device: T.device,
) -> tuple[list[T.Tensor], list[T.Tensor]]:
    X, Y = create_dataset(device)
    partitioned_X = T.chunk(X, NUM_PEAKS)
    partitioned_Y = T.chunk(Y, NUM_PEAKS)

    return partitioned_X, partitioned_Y


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    kan = KAN(
        KAN_ARCHITECTURE,
        KAN_GRID_SIZE,
        grid_range=[0, NUM_PEAKS],
        bias_trainable=False,
        sp_trainable=False,
        sb_trainable=False,
        device=device,
    )
    mlp = MLP(MLP_ARCHITECTURE).to(device)

    writer = ExperimentWriter(EXPERIMENT_NAME)

    X, Y = create_dataset(device)
    X_partitioned, Y_partitioned = create_partitioned_dataset(device)

    fig, ax = plt.subplots()
    ax.plot(X.cpu(), Y.cpu(), color="black")
    writer.log_graph("training_function", fig)

    fig, ax = plt.subplots(1, NUM_PEAKS, figsize=(15, 2))
    for i, (x, y) in enumerate(zip(X_partitioned, Y_partitioned)):
        ax[i].plot(x.cpu(), y.cpu(), color="black")
        ax[i].plot(X.cpu(), Y.cpu(), color="black", alpha=0.1)
    writer.log_graph("partitioned_function", fig)

    kan_preds = []
    mlp_preds = []

    for i, (x, y) in tqdm(
        enumerate(zip(X_partitioned, Y_partitioned)), total=NUM_PEAKS
    ):
        dataset = {
            "train_input": x,
            "train_label": y,
            "test_input": x,
            "test_label": y,
        }

        kan.train(
            dataset,
            steps=NUM_EPOCHS,
            device=device,
            update_grid=False,
            disable_pbar=True,
        )
        train_model(mlp, dataset, NUM_EPOCHS)

        with T.no_grad():
            kan_preds.append(kan(X))
            mlp_preds.append(mlp(X))

    fig, ax = plt.subplots(3, NUM_PEAKS, figsize=(15, 2))
    for i, (kan_pred, mlp_pred) in enumerate(zip(kan_preds, mlp_preds)):
        ax[0][i].plot(X_partitioned[i].cpu(), Y_partitioned[i].cpu(), color="black")
        ax[0][i].plot(X.cpu(), Y.cpu(), color="black", alpha=0.1)
        ax[0][i].set_ylim(-0.5, 1.5)

        ax[1][i].plot(X.cpu(), kan_pred.cpu(), color="black")
        ax[1][i].plot(X.cpu(), Y.cpu(), color="black", alpha=0.1)
        ax[1][i].set_ylim(-0.5, 1.5)

        ax[2][i].plot(X.cpu(), mlp_pred.cpu(), color="black")
        ax[2][i].plot(X.cpu(), Y.cpu(), color="black", alpha=0.1)
        ax[2][i].set_ylim(-0.5, 1.5)

    writer.log_graph("training_graphs", fig)

    writer.write()


if __name__ == "__main__":
    main()
