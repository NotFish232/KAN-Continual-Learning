import torch as T
from pathlib import Path
import pandas as pd

MNIST_TRAIN_PATH = "./datasets/MNIST/mnist_train.csv"
MNIST_TEST_PATH = "./datasets/MNIST/mnist_test.csv"
IMG_SIZE = 28

EXAMPLE_NAME = Path(__file__).stem


def load_dataset(
    path: str,
    device: T.device,
) -> tuple[list[T.Tensor], list[T.Tensor]]:
    data = T.tensor(pd.read_csv(path).to_numpy(), device=device)
    data[:, 1:] = data[:, 1:] / 255

    batched_images = []
    batched_labels = []

    for i in range(10):
        indices = data[:, 0] == i
        batched_images.append(data[indices, 1:])
        batched_labels.append(data[indices, :1])

    return batched_images, batched_labels


def main() -> None:
    device = T.device("cpu")

    X_train, Y_train = load_dataset(MNIST_TRAIN_PATH, device)
    X_test, Y_test = load_dataset(MNIST_TEST_PATH, device)



if __name__ == "__main__":
    main()
