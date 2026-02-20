import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation


class PyramidPool2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        bins_size: list[int] = [1, 2, 3, 6],
    ):
        """Pyramid pooling 2D [1612.01105v2]

        Args:
            input_channel (int): input channel size
            bins_size (list[int], optional): size of pooled feature maps. Defaults to [1, 2, 3, 6].
        """
        super().__init__()
        self.input_channel = input_channel
        self.bins_size = bins_size
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    Conv2dNormActivation(
                        input_channel,
                        input_channel // len(bins_size),
                        1,
                    ),
                )
                for bin_size in bins_size
            ]
        )

    @property
    def output_channel(self) -> int:
        return self.input_channel * (1 + len(self.bins_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for branch in self.branches:
            feature = branch(x)
            feature = F.interpolate(
                feature,
                x.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            features.append(feature)

        return torch.cat(features, dim=1)
