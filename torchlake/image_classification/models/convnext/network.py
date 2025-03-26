import torch
from torch import nn


class BottleNeck(nn.Module):

    def __init__(self, base_channel: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 7, padding=3, groups=base_channel),
            nn.GroupNorm(1, base_channel),
            nn.Conv2d(base_channel, base_channel * 4, 1),
            nn.GELU(),
            nn.Conv2d(base_channel * 4, base_channel, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
