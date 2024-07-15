from pathlib import Path

import pandas as pd
import torch as T
from kan import KAN
from tqdm import tqdm
import numpy as np

from utils import (
    suggest_KAN_architecture,
    suggest_MLP_architecture,
    train_model,
    num_parameters,
)
from utils.io import ExperimentWriter
from utils.models import MLP

EXPERIMENT_NAME = Path(__file__).parent.name

MNIST_TRAIN_PATH = "./data/MNIST/mnist_train.csv"
MNIST_TEST_PATH = "./data/MNIST/mnist_test.csv"


IMG_SIZE = 28


NUM_KAN_EPOCHS = 5
NUM_MLP_EPOCHS = 500
NUM_PARAMETERS = 10_000

MLP_ARCHITECTURE = [IMG_SIZE**2, 128, 128, 64, 10]
KAN_ARCHITECTURE = [IMG_SIZE**2, 32, 10]
KAN_GRID_SIZE = 32


def load_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    train_dataset = pd.read_csv(MNIST_TRAIN_PATH)
    test_dataset = pd.read_csv(MNIST_TEST_PATH)


def load_dataset(
    dataset_path: str, device: T.device
) -> tuple[list[T.Tensor], list[T.Tensor]]:
    dataset = pd.read_csv(MNIST_TRAIN_PATH).to_numpy()
    X = dataset[:, 1:].astype(np.float32) / 255
    Y = dataset[:, :1]
 
    split_X = []
    split_Y = [] 

    for i, j in zip(range(0, 10, 2), range(1, 10, 2)):
        indices = ((Y == i) | (Y == j)).squeeze()
        split_X.append(X[indices])
        split_Y.append(Y[indices])
    
    return split_X, split_Y

def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    kan = KAN(
        KAN_ARCHITECTURE,
        KAN_GRID_SIZE,
        device=device,
    )
    mlp = MLP(MLP_ARCHITECTURE).to(device)

    writer = ExperimentWriter(EXPERIMENT_NAME)
    print(num_parameters(kan))
    X_train, Y_train = load_dataset(MNIST_TRAIN_PATH, device)

    exit(1)
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
