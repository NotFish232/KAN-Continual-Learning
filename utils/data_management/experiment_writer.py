import pickle
from datetime import datetime
from typing import Any

import torch as T
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT, ExperimentDataType


class ExperimentWriter:
    """
    Helper class to write experiments in a nice parsable way
    """

    def __init__(
        self: Self, experiment_name: str, experiment_dtype: ExperimentDataType
    ) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name
        self.experiment_dtype = experiment_dtype

        self.config: dict[str, Any] = {}
        self.data: dict[str, list[T.Tensor] | T.Tensor] = {}

    def log_config(self: Self, name: str, data: Any) -> None:
        self.config[name] = data

    def log_data(self: Self, name: str, data: list[T.Tensor] | T.Tensor) -> None:
        if isinstance(data, list):
            self.data[name] = [d.detach().cpu() for d in data]
        else:
            self.data[name] = data.detach().cpu()

    def write(self: Self) -> None:
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        filename = f"run_{datetime.now():%Y_%m_%d__%H_%M_%S}.pickle"

        full_data = {
            "experiment_name": self.experiment_name,
            "experiment_dtype": self.experiment_dtype.value,
            "config": self.config,
            "data": self.data,
        }

        with open(self.experiment_path / filename, "wb+") as f:
            pickle.dump(full_data, f, pickle.HIGHEST_PROTOCOL)
