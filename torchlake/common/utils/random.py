import random

import torch


def rand_color(maximum: int = 256, channel: int = 3) -> list[int]:
    return random.choices(range(maximum), k=channel)


def generate_normal(*size: torch.Size, mu: float = 0, sd: float = 1) -> torch.Tensor:
    return torch.empty(size).normal_(mu, sd)
