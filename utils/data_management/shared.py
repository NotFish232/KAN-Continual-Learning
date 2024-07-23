from enum import Enum
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"


class ExperimentDataType(Enum):
    """
    Enum representing the data that an experiment is trained on
    * function_1d => a curve
    * function_2d => a surface
    * image_bw => black white image data, i.e. mnist
    * image_rgb => three channel image data, i.e. cifar100
    * none => no experiment data type
    """
    function_1d = 1
    function_2d = 2
    image_bw = 3
    image_rgb = 4
    none = 5