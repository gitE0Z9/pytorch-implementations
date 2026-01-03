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
            # B, C * patch_size, #patch
            F.unfold(x, patch_shape, stride=patch_shape)
            # B, C, patch_h, patch_w, #patch
            .unflatten(1, (c, *patch_shape))
            # B, #patch, C, patch_h, patch_w
            .permute(0, 4, 1, 2, 3)
            # B, #patch * C, patch_h, patch_w
            .flatten(1, 2)
        )
