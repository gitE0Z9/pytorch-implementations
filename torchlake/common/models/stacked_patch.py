import torch
import torch.nn.functional as F
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
        _, c, h, w = x.shape
        k = (h // self.stride, w // self.stride)
        return (
            F.unfold(x, k, stride=k)
            .unflatten(1, (c, *k))
            .permute(0, 1, 4, 2, 3)
            .flatten(1, 2)
        )
