import random

import torch


def rand_color(maximum: int = 256, channel: int = 3) -> list[int]:
    return random.choices(range(maximum), k=channel)


def generate_normal(*size: int, mu: float = 0, sd: float = 1) -> torch.Tensor:
    return torch.randn(*size) * sd + mu


def generate_uniform(*size: int, a: float = 0, b: float = 1) -> torch.Tensor:
    return torch.rand(*size) * (b - a) + a
