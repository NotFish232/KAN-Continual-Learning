from pathlib import Path

import pandas as pd
import torch as T
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

from utils.architecture import KAN_ARCHITECTURE, MLP_ARCHITECTURE
from utils.data_management import ExperimentDataType
from utils.experiment import run_experiment
from utils.training import TrainModelArguments, calculate_accuracy

EXPERIMENT_NAME = Path(__file__).stem

CIFAR_TRAIN_PATH = "./data/CIFAR_100/cifar_100_train.csv"
CIFAR_EVAL_PATH = "./data/CIFAR_100/cifar_100_eval.csv"
CIFAR_LABELS_PATH = "./data/CIFAR_100/cifar_100_labels.txt"
IMG_SIZE = 32

NUM_LABELS = 100
NUM_TASKS = 20
LABELS_PER_TASK = NUM_LABELS // NUM_TASKS


PARAMETER_COUNTS = [50_000, 200_000, 1_000_000]


NUM_EPOCHS = 1
LR = 2e-3

# number of image samples used for calculating metrics per task
NUM_TASK_EVAL_SAMPLES = 50

# number of images samples used for predictions per task
NUM_TASK_PREDICTION_SAMPLES = 50

EVAL_BATCH_SIZE = 10


def load_task_datasets(path: str, device: T.device) -> list[Dataset]:
    data = pd.read_csv(path, header=None).to_numpy()

    X = T.tensor(data[:, 1:], dtype=T.float32, device=device) / 255
    Y = T.eye(NUM_LABELS, device=device)[T.asarray(data[:, 0])]

    datasets: list[Dataset] = []

    for task in range(NUM_TASKS):
        indices = T.zeros(
            (len(Y),),
            dtype=T.bool,
            device=device,
        )
        for label in range(task * LABELS_PER_TASK, (task + 1) * LABELS_PER_TASK):
            indices |= Y.argmax(dim=-1) == label

        x_batch = X[indices]
        y_batch = Y[indices]

        datasets.append(TensorDataset(x_batch, y_batch))

    return datasets


def main() -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    train_datasets = load_task_datasets(CIFAR_TRAIN_PATH, device)
    full_eval_datasets = load_task_datasets(CIFAR_EVAL_PATH, device)

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
        [
            (KAN_ARCHITECTURE[(IMG_SIZE**2 * 3, NUM_LABELS)][p], p)
            for p in PARAMETER_COUNTS
        ],
        [
            (MLP_ARCHITECTURE[(IMG_SIZE**2 * 3, NUM_LABELS)][p], p)
            for p in PARAMETER_COUNTS
        ],
        train_datasets,
        eval_datasets,
        prediction_datasets,
        prediction_ground_truths,
        ExperimentDataType.image_rgb,
        device=device,
        kan_kwargs={
            "bias_trainable": False,
            "sp_trainable": False,
            "sb_trainable": False,
        },
        mlp_kwargs={"activation_fn": F.leaky_relu},
        training_args=TrainModelArguments(
            num_epochs=NUM_EPOCHS,
            lr=LR,
            eval_batch_size=EVAL_BATCH_SIZE,
            loss_fn=nn.CrossEntropyLoss(),
            eval_fns={"loss": nn.CrossEntropyLoss(), "acc": calculate_accuracy},
        ),
    )


if __name__ == "__main__":
    main()
