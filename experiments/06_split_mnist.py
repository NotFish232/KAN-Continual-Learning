from pathlib import Path

import pandas as pd
import torch as T
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

from utils.data_management import ExperimentDataType
from utils.experiment import run_experiment
from utils.training import TrainModelArguments, calculate_accuracy

EXPERIMENT_NAME = Path(__file__).stem

MNIST_TRAIN_PATH = "./data/MNIST/mnist_train.csv"
MNIST_EVAL_PATH = "./data/MNIST/mnist_eval.csv"
IMG_SIZE = 28


KAN_ARCHITECTURE = [IMG_SIZE**2, 16, 10]
KAN_GRID_SIZE = 25
MLP_ARCHICTURE = [IMG_SIZE**2, 128, 128, 64, 10]

NUM_EPOCHS = 1

# number of image samples used for calculating metrics per task
NUM_TASK_EVAL_SAMPLES = 10

# number of images samples used for predictions per task
NUM_TASK_PREDICTION_SAMPLES = 10


def load_task_datasets(path: str, device: T.device) -> list[Dataset]:
    data = pd.read_csv(path).to_numpy()

    X = T.tensor(data[:, 1:], dtype=T.float32, device=device) / 255
    Y = T.eye(10, device=device)[T.asarray(data[:, 0])]

    datasets: list[Dataset] = []

    for label_1, label_2 in zip(range(0, 11, 2), range(1, 11, 2)):
        indices = (
            (Y.argmax(dim=-1) == label_1) | (Y.argmax(dim=-1) == label_2)
        ).squeeze()
        x_batch = X[indices]
        y_batch = Y[indices]

        datasets.append(TensorDataset(x_batch, y_batch))

    return datasets


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    train_datasets = load_task_datasets(MNIST_TRAIN_PATH, device)
    full_eval_datasets = load_task_datasets(MNIST_EVAL_PATH, device)

    eval_datasets: dict[str, list[Dataset] | Dataset] = {}
    prediction_datasets: dict[str, list[T.Tensor] | T.Tensor] = {}
    prediction_ground_truths: dict[str, list[T.Tensor] | T.Tensor] = {}

    eval_per_task_datasets: list[Dataset] = []
    prediction_per_task_datasets: list[T.Tensor] = []
    ground_truths_per_task: list[T.Tensor] = []

    for dataset in full_eval_datasets:
        eval_per_task_datasets.append(Subset(dataset, range(NUM_TASK_EVAL_SAMPLES)))

        prediction_dataset, ground_truths = dataset[range(NUM_TASK_PREDICTION_SAMPLES)]
        prediction_per_task_datasets.append(prediction_dataset)
        ground_truths_per_task.append(ground_truths)

    eval_datasets["eval"] = eval_per_task_datasets
    prediction_datasets["eval"] = prediction_per_task_datasets
    prediction_ground_truths["eval"] = ground_truths_per_task

    eval_datasets["whole"] = ConcatDataset(eval_per_task_datasets)
    prediction_datasets["whole"] = T.concat(prediction_per_task_datasets)
    prediction_ground_truths["whole"] = T.concat(ground_truths_per_task)

    run_experiment(
        EXPERIMENT_NAME,
        KAN_ARCHITECTURE,
        MLP_ARCHICTURE,
        train_datasets,
        eval_datasets,
        prediction_datasets,
        prediction_ground_truths,
        ExperimentDataType.image,
        kan_kwargs={"grid": KAN_GRID_SIZE},
        mlp_kwargs={"activation_fn": F.leaky_relu},
        training_args=TrainModelArguments(
            num_epochs=NUM_EPOCHS,
            loss_fn=nn.CrossEntropyLoss(),
            eval_fns={"loss": nn.CrossEntropyLoss(), "acc": calculate_accuracy},
        ),
    )


if __name__ == "__main__":
    main()
