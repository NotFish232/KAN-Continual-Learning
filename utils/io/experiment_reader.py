import pickle

from natsort import natsorted
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT
import torch as T


class ExperimentReader:
    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name
        self.filename = natsorted(self.experiment_path.iterdir())[-1]

        self.data: dict[str, T.Tensor] = {}

    def read(self: Self) -> None:
        with open(self.experiment_path / self.filename, "rb") as f:
            self.data = pickle.load(f)

    

