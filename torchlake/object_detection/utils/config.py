from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f.read())


def load_classes(path: str) -> list:
    with open(path, "r") as f:
        return f.read().split("\n")


def load_anchors(anchors_path: str | Path) -> torch.Tensor:
    """load anchors from file

    Args:
        anchors_path (str | Path): path to anchors file

    Returns:
        torch.Tensor: anchors, in format of (cx, cy, w, h), in shape of (#num_anchor * #num_grid_y * #num_grid_x, 4)
    """
    anchors_path = Path(anchors_path)
    anchors = np.loadtxt(anchors_path, delimiter=",")
    anchors = torch.from_numpy(anchors).float()

    return anchors


def save_anchors(anchors_path: str | Path, anchors: np.ndarray):
    """save anchors to file

    Args:
        anchors_path (str | Path): path to anchors file
        anchors (np.ndarray): anchors, in format of (cx, cy, w, h), in shape of (#num_anchor * #num_grid_y * #num_grid_x, 4)
    """
    with anchors_path.open("w") as f:
        for cx, cy, w, h in anchors.tolist():
            print(f"{cx},{cy},{w},{h}", file=f)
