import torch
import torchvision.transforms.functional as F
from torch import nn


class ModCrop(nn.Module):
    def __init__(self, scale: int):
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, "must be rank-4 tensor"

        target_shape = tuple(s - s % self.scale for s in x[2:])

        return F.crop(x, 0, 0, *target_shape)
