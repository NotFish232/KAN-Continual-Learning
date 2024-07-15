from pathlib import Path

import torch as T
from natsort import natsorted
from plotly import graph_objects as go  # type: ignore

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"


def get_experiments() -> list[str]:
    return natsorted(p.name for p in Path(EXPERIMENT_ROOT).iterdir() if p.is_dir())


def get_experiment_plots(
    experiment_name: str, experiment_data: dict[str, T.Tensor]
) -> dict[str, go.Figure]:
    experiment_visualizer = __import__(f"experiments.{experiment_name}.visualize", fromlist=[None])  # type: ignore
    return experiment_visualizer.create_plots(experiment_data)
