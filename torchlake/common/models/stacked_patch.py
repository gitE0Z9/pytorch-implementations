import torch
from torch import nn


class StackedPatch2d(nn.Module):
    def __init__(self, stride: int):
        """Stack patch

        Args:
            stride (int): stride
        """
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(
            b,
            c * self.stride * self.stride,
            h // self.stride,
            w // self.stride,
        )
        return x
