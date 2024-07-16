from pathlib import Path

import torch as T
from natsort import natsorted
from plotly import graph_objects as go  # type: ignore

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"

