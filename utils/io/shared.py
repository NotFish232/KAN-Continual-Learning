from enum import Enum
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parents[2] / "results"


class LogType(Enum):
    graph = 0
    data = 1


def get_experiments() -> list[str]:
    return [p.name for p in Path(EXPERIMENT_ROOT).iterdir() if p.is_dir()]


