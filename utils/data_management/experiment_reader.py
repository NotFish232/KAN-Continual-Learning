import pickle
from pathlib import Path
from typing import Any

import torch as T
from natsort import natsorted
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT, ExperimentDataType


class ExperimentReader:
    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_dtype = ExperimentDataType.none

        self.experiment_path = EXPERIMENT_ROOT / experiment_name

        potential_filenames = natsorted(self.experiment_path.iterdir(), reverse=True)
        self.filename = next(iter(potential_filenames), None)

        self.config: dict[str, Any] = {}
        self.data: dict[str, list[T.Tensor] | T.Tensor] = {}

    def read(self: Self) -> None:
        if self.filename is None:
            return

        with open(self.experiment_path / self.filename, "rb") as f:
            full_data = pickle.load(f)

        self.experiment_name = full_data["experiment_name"]
        self.experiment_dtype = full_data["experiment_dtype"]
        self.config = full_data["config"]
        self.data = full_data["data"]

    @staticmethod
    def get_experiments() -> list[str]:
        return natsorted(p.name for p in Path(EXPERIMENT_ROOT).iterdir() if p.is_dir())
