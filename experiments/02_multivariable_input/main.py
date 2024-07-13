import math
from pathlib import Path

import torch as T
from kan import KAN
from tqdm import tqdm

from utils import (
    suggest_KAN_architecture,
    suggest_MLP_architecture,
    train_model,
    gaussian,
)
from utils.io import ExperimentWriter
from utils.models import MLP

EXPERIMENT_NAME = Path(__file__).parent.name

NUM_PEAKS = 2
LINEAR_SLOPE = 0.25
NUM_POINTS = 100
GAUSSIAN_STD_1 = 0.2
GAUSSIAN_STD_2 = 0.1

NUM_KAN_EPOCHS = 5
NUM_MLP_EPOCHS = 500
NUM_PARAMETERS = 1_000

MLP_ARCHITECTURE = suggest_MLP_architecture(
    num_inputs=2,
    num_outputs=1,
    num_layers=3,
    num_params=NUM_PARAMETERS,
)
KAN_ARCHITECTURE, KAN_GRID_SIZE = suggest_KAN_architecture(
    num_inputs=2,
    num_outputs=1,
    num_layers=1,
    num_params=NUM_PARAMETERS,
)


def create_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(0, NUM_PEAKS, NUM_POINTS, device=device)
    xy = T.cartesian_prod(x, x)
    z = T.zeros((NUM_POINTS**2, 1), device=device)
    for i in range(NUM_PEAKS):
        for j in range(NUM_PEAKS):
            z += T.cartesian_prod(
                gaussian(x, i + 0.5, GAUSSIAN_STD_1),
                gaussian(x, j + 0.5, GAUSSIAN_STD_2),
            ).sum(dim=-1, keepdim=True)

    # evil permuting to create tensor that is partitioned into n square shaped domain tasks
    xy = (
        xy.reshape(*([int(math.sqrt(NUM_POINTS))] * 4), 2)
        .permute(0, 2, 1, 3, 4)
        .reshape(-1, 2)
    )
    z = (
        z.reshape(*([int(math.sqrt(NUM_POINTS))] * 4), 1)
        .permute(0, 2, 1, 3, 4)
        .reshape(-1, 1)
    )

    return xy, z


def create_partitioned_dataset(
    device: T.device,
) -> tuple[T.Tensor, T.Tensor]:
    X, Y = create_dataset(device)
    X_partitioned = T.cat([x.unsqueeze(0) for x in T.chunk(X, NUM_PEAKS**2)])
    Y_partitioned = T.cat([y.unsqueeze(0) for y in T.chunk(Y, NUM_PEAKS**2)])

    return X_partitioned, Y_partitioned


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
    writer.log("X", X)
    writer.log("Y", Y)
    writer.log("X_partitioned", X_partitioned)
    writer.log("Y_partitioned", Y_partitioned)

    kan_preds = []
    mlp_preds = []
    kan_train_loss: list[float] = []
    kan_test_loss: list[float] = []
    mlp_train_loss: list[float] = []
    mlp_test_loss: list[float] = []

    for i, (x, y) in tqdm(
        enumerate(zip(X_partitioned, Y_partitioned)), total=NUM_PEAKS**2
    ):
        dataset = {
            "train_input": x,
            "train_label": y,
            "test_input": X,
            "test_label": Y,
        }

        kan_results = kan.train(
            dataset,
            steps=NUM_KAN_EPOCHS,
            device=device,
            update_grid=False,
            disable_pbar=True,
        )
        mlp_results = train_model(mlp, dataset, NUM_MLP_EPOCHS)

        kan_train_loss.extend(l.item() for l in kan_results["train_loss"])
        kan_test_loss.extend(l.item() for l in kan_results["test_loss"])
        mlp_train_loss.extend(l.item() for l in mlp_results["train_loss"])
        mlp_test_loss.extend(l.item() for l in mlp_results["test_loss"])

        with T.no_grad():
            kan_preds.append(kan(X))
            mlp_preds.append(mlp(X))

    writer.log("kan_preds", T.cat([p.unsqueeze(0) for p in kan_preds]))
    writer.log("mlp_preds", T.cat([p.unsqueeze(0) for p in mlp_preds]))
    writer.log("kan_train_loss", T.tensor(kan_train_loss))
    writer.log("kan_test_loss", T.tensor(kan_test_loss))
    writer.log("mlp_train_loss", T.tensor(mlp_train_loss))
    writer.log("mlp_test_loss", T.tensor(mlp_test_loss))

    writer.write()


if __name__ == "__main__":
    main()
