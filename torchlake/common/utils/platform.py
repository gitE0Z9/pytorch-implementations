import os
import platform
from pathlib import Path
from typing import Literal


def get_num_workers() -> int:
    if platform.system() == "Windows":
        return 0
    else:
        return os.cpu_count()


def get_file_size(
    path: str | Path,
    unit: Literal["K"] | Literal["M"] | Literal["G"] = "M",
) -> float:
    if isinstance(path, str):
        path = Path(path)

    mapping = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }
    divider = mapping[unit]

    return f"{round(path.stat().st_size / divider, 2)}{unit}iB"
