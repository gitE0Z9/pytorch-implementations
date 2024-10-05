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
        super(PyramidPool2d, self).__init__()
        self.layers = nn.ModuleList(
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [x]
        for layer in self.layers:
            feature = layer(x)
            feature = F.interpolate(feature, x.shape[2:], mode="bilinear")
            features.append(feature)

        return torch.cat(features, dim=1)
