import torch
import yaml
from numpy import loadtxt


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f.read())


def load_classes(path: str) -> list:
    with open(path, "r") as f:
        return f.read().split("\n")


def load_anchors(path: str) -> torch.Tensor:
    anchors = loadtxt(path, delimiter=",")
    anchors = torch.from_numpy(anchors).float().view(1, len(anchors), 2, 1, 1)

    return anchors  # 1, 5, 2, 1, 1
