import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation


class PyramidPool2d(nn.Module):

    def __init__(
        self,
        in_dim: int,
        bins_size: list[int] = [1, 2, 3, 6],
    ):
        super(PyramidPool2d, self).__init__()
        self.kernel_size = bins_size
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    Conv2dNormActivation(in_dim, in_dim // len(bins_size), 1),
                )
                for bin_size in bins_size
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [layer(x) for layer in self.layers]
        features = [
            F.interpolate(
                feature,
                x.shape[2:],
                mode="bilinear",
            )
            for feature in features
        ]
        return torch.cat([x, *features], dim=1)
