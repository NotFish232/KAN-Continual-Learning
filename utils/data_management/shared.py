from enum import Enum
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"


class ExperimentDataType(Enum):
    """
    Enum representing the data that an experiment is trained on
    * function_1d => a curve
    * function_2d => a surface
    * image => image data, i.e. mnist
    * none => no experiment data type
    """
    function_1d = 1
    function_2d = 2
    image = 3
    none = 4