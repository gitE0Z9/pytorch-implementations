import random

import torch


def rand_color(maximum: int = 256, channel: int = 3) -> list[int]:
    return random.choices(range(maximum), k=channel)


def generate_standard_normal(*size: torch.Size) -> torch.Tensor:
    return torch.empty(size).normal_(0, 1)
