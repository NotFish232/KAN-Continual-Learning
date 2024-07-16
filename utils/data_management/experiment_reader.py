import pickle

import torch as T
from natsort import natsorted
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT
from pathlib import Path


class ExperimentReader:
    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name
        self.filename = natsorted(self.experiment_path.iterdir())[-1]

        self.data: dict[str, list[T.Tensor] | T.Tensor] = {}

    def read(self: Self) -> None:
        with open(self.experiment_path / self.filename, "rb") as f:
            self.data = pickle.load(f)
    
    @staticmethod
    def get_experiments() -> list[str]:
        return natsorted(p.name for p in Path(EXPERIMENT_ROOT).iterdir() if p.is_dir())



    

