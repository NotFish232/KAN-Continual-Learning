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

NUM_PEAKS = 5
NUM_POINTS = 500
GAUSSIAN_STD = 0.2

NUM_KAN_EPOCHS = 5
NUM_MLP_EPOCHS = 500
NUM_PARAMETERS = 100

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


def create_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(0, NUM_PEAKS, NUM_POINTS, device=device).unsqueeze(1)
    y = T.zeros_like(x)
    for i in range(NUM_PEAKS):
        y += gaussian(x, i + 0.5, GAUSSIAN_STD)

    return x, y


def create_partitioned_dataset(
    device: T.device,
) -> tuple[T.Tensor, T.Tensor]:
    X, Y = create_dataset(device)
    X_partitioned = T.cat([x.unsqueeze(0) for x in T.chunk(X, NUM_PEAKS)])
    Y_partitioned = T.cat([y.unsqueeze(0) for y in T.chunk(Y, NUM_PEAKS)])

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
        enumerate(zip(X_partitioned, Y_partitioned)), total=NUM_PEAKS
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
