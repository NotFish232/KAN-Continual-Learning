from typing_extensions import Self
import torch as T
from typing import Any
from .shared import EXPERIMENT_ROOT


class ExperimentWriter:
    """
    Helper class to write experiments in a nice parsable way
    """

    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name

        self.data: list[dict[str, Any]] = []

    def log_graph(self: Self, graph_name: str, x: T.Tensor, y: T.Tensor) -> None:
        self.data.append({"type": "graph", "name": graph_name, "x": x, "y": y})

    def write(self: Self) -> None:
        print(self.experiment_path)
