import torch
from torch import nn


class SqueezeExcitation2d(nn.Module):

    def __init__(self, in_dim: int, reduction_ratio: float = 1):
        super(SqueezeExcitation2d, self).__init__()
        self.s = nn.Conv2d(in_dim, in_dim // reduction_ratio, 1)
        self.e = nn.Conv2d(in_dim // reduction_ratio, in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean((2, 3), keepdim=True)
        y = self.s(y)
        y.relu_()
        y = self.e(y)
        return x * y.sigmoid()
