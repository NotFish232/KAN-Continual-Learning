import pickle
from datetime import datetime
import torch as T
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT


class ExperimentWriter:
    """
    Helper class to write experiments in a nice parsable way
    """

    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name

        self.data: dict[str, T.Tensor] = {}

    def log(self: Self, name: str, data: T.Tensor) -> None:
        self.data[name] = data.detach().cpu()

    def write(self: Self) -> None:
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        filename = f"run_{datetime.now():%Y_%m_%d__%H_%M_%S}.pickle"

        with open(self.experiment_path / filename, "wb+") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
