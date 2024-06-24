from enum import Enum

from .b_spline import BSplineKanLayer
from .fourier_series import FourierSeriesKanLayer


class KanLayer(Enum):
    b_spline = BSplineKanLayer
    fourier_series = FourierSeriesKanLayer