import torch as T
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Self


MNIST_TRAIN_PATH = "./datasets/MNIST/mnist_train.csv"
MNIST_TEST_PATH = "./datasets/MNIST/mnist_test.csv"

class DatasetType(Enum):
    training = 0
    testing = 1

class MnistDataset(Dataset):
    def __init__(self: Self, dataset_type: DatasetType) -> None:
        self.data