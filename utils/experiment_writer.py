from pathlib import Path

from typing_extensions import Self

EXPERIMENT_ROOT = Path(__file__).parent / "results"


class ExperimentWriter:
    """
    Helper class to write experiments in a nice parsable way
    """

    def __init__(self: Self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
