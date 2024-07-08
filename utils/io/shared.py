from pathlib import Path
from enum import Enum

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"


class LogType(Enum):
    graph = 0


def get_experiments() -> list[str]:
    return [p.name for p in Path(EXPERIMENT_ROOT).iterdir() if p.is_dir()]


