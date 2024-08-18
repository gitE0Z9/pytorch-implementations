import torch
from torch import nn


class FlattenFeature(nn.Module):

    def __init__(self):
        super(FlattenFeature, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
