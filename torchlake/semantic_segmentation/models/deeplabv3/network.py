import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models import FlattenFeature
from ....image_classification.models.resnet import BottleNeckB, ResBlock


class ASPP(nn.Module):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_channel: int,
        dilations: list[int],
    ):
        """A'trous spatial pyramid pooling in paper [1706.05587v3]

        Args:
            input_channel (int): input channel size of dilated conlutions
            hidden_dim (int): output channel size of dilated conlutions
            output_channel (int): output channel size of last 1x1 conv
            dilations (list[int]): dilation size of ASPP, for 16x [6, 12, 18], for 8x [12, 24, 36].
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv2dNormActivation(input_channel, hidden_dim, 1),
                *[
                    Conv2dNormActivation(
                        input_channel,
                        hidden_dim,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    )
                    for dilation in dilations
                ],
                nn.Sequential(
                    FlattenFeature(end_dim=1),
                    Conv2dNormActivation(input_channel, hidden_dim, 1),
                ),
            ]
        )
        self.head = Conv2dNormActivation(
            (2 + len(dilations)) * hidden_dim,
            output_channel,
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [block(x) for block in self.blocks]
        y[-1] = F.interpolate(
            y[-1],
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        y = torch.cat(y, 1)

        return self.head(y)


class CascadeASPP(nn.Module):

    def __init__(self, dilations: list[int] = [8, 16, 32]):
        """Cascade ASPP in paper [1706.05587v3]

        Args:
            dilations (list[int]): dilation size of ASPP. Defaults is [8, 16, 32].
        """
        super().__init__()
        # block 5, 6, 7
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    *[ResBlock(2048, 512, 2048, block=BottleNeckB) for _ in range(3)]
                )
                for _ in range(3)
            ]
        )
        for i, block in enumerate(self.blocks):
            for key, layer in block.named_modules():
                if "conv2" in key:
                    scale = dilations[i]
                    layer.dilation, layer.padding = (scale, scale), (scale, scale)
                elif "downsample.0" in key:
                    layer.stride = (1, 1)
        self.head = Conv2dNormActivation(2048, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        return self.head(y)
