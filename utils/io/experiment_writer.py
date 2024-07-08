import pickle
from datetime import datetime
from typing import Any

import torch as T
from matplotlib.figure import Figure
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT, LogType


class ExperimentWriter:
    """
    Helper class to write experiments in a nice parsable way
    """

    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name

        self.data: list[dict[str, Any]] = []

    def log_graph(self: Self, graph_name: str, graph: Figure) -> None:
        self.data.append(
            {
                "type": LogType.graph,
                "name": graph_name,
                "data": graph,
            }
        )

    def write(self: Self) -> None:
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        filename = f"run_{datetime.now():%Y_%m_%d__%H_%M_%S}.pickle"

        with open(self.experiment_path / filename, "wb+") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
