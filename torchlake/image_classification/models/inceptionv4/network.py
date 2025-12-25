import torch
from torch import nn


class ActivationScaling(nn.Module):
    def __init__(self, scale: float = 0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale
