from enum import Enum
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"


class ExperimentDataType(Enum):
    function_1d = 1
    function_2d = 2
    image = 3
    none = 4