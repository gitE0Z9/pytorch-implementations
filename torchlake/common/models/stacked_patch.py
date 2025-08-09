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
        patch_shape = (h // self.stride, w // self.stride)
        return (
            # B, C * #patch, self.stride**2
            F.unfold(x, patch_shape, stride=patch_shape)
            # B, C, #patch_v, #patch_h, self.stride**2
            .unflatten(1, (c, *patch_shape))
            # B, self.stride**2, C, #patch_v, #patch_h
            .permute(0, 4, 1, 2, 3)
            # B, self.stride**2 * C, #patch_v, #patch_h
            .flatten(1, 2)
        )
