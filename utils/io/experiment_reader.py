import pickle
from typing import Any, Generator

from natsort import natsorted
from typing_extensions import Self

from .shared import EXPERIMENT_ROOT


class ExperimentReader:
    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_path = EXPERIMENT_ROOT / experiment_name
        self.filename = natsorted(self.experiment_path.iterdir())[-1]

        self.data: list[dict[str, Any]] = []

    def read(self: Self) -> Generator[dict[str, Any], None, None]:
        with open(self.experiment_path / self.filename, "rb") as f:
            self.data = pickle.load(f)
        
        yield from self.data

    

