import torch
from torch import nn
from .constants import IMAGENET_MEAN, IMAGENET_STD


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


class Normalization(nn.Module):
    def __init__(
        self,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ):
        super(Normalization, self).__init__()
        ## C,1,1 shape for broadcasting
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        original_shape = img.size()

        if img.dim() != 3:
            img = img.reshape(-1, img.size(-2), img.size(-1))

        normalized = (img - self.mean.to(img.device)) / self.std.to(img.device)

        return normalized.reshape(*original_shape)
