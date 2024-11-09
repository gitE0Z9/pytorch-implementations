from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    if isinstance(path, str):
        path = Path(path)

    return yaml.safe_load(path.read_text())


def load_classes(path: str | Path) -> list:
    if isinstance(path, str):
        path = Path(path)

    return path.read_text().splitlines()
