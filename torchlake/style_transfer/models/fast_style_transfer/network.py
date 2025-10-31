import torch
from torch import nn

from torchlake.common.models import ConvINReLU


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvINReLU(hidden_dim, hidden_dim, 3),
            ConvINReLU(hidden_dim, hidden_dim, 3, activation=None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
