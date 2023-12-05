import torch
from torch import nn


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int,
        s: int = 1,
        p: int = 0,
        d: int = 1,
        g: int = 1,
        bn: bool = True,
        relu: bool = True,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, d, g, bias=not bn)
        self.bn = nn.BatchNorm2d(out_c) if bn else bn
        self.relu = nn.ReLU(True) if relu else relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        if self.relu:
            y = self.relu(y)

        return y
