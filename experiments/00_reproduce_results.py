from kan import KAN
import torch as T

from utils import num_parameters, suggest_KAN_architecture, suggest_MLP_architecture
from utils.io import ExperimentWriter
from utils.models import MLP
from pathlib import Path
from matplotlib import pyplot as plt


EXPERIMENT_NAME = Path(__file__).stem

NUM_PEAKS = 5
NUM_POINTS = 500
GAUSSIAN_STD = 0.2

NUM_PARAMETERS = 5_000

MLP_ARCHITECTURE = suggest_MLP_architecture(
    num_inputs=1,
    num_outputs=1,
    num_layers=4,
    num_params=NUM_PARAMETERS,
)
KAN_ARCHITECTURE, KAN_GRID_SIZE = suggest_KAN_architecture(
    num_inputs=1,
    num_outputs=1,
    num_layers=1,
    num_params=NUM_PARAMETERS,
)


def guassian(x: T.Tensor, mean: float, std: float) -> T.Tensor:
    return T.exp(-((x - mean) ** 2) / (2 * std**2))


def create_dataset(device: T.device) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(0, NUM_PEAKS, NUM_POINTS, device=device).unsqueeze(1)
    y = T.zeros_like(x)
    for i in range(NUM_PEAKS):
        y += guassian(x, i + 0.5, GAUSSIAN_STD)

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
    models = [
        ("MLP", MLP(MLP_ARCHITECTURE)),
        ("KAN", KAN(KAN_ARCHITECTURE, KAN_GRID_SIZE)),
    ]

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

    writer.write()


if __name__ == "__main__":
    main()
