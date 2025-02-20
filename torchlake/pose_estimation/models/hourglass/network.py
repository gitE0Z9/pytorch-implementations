import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.models import ResBlock

from torchlake.common.models.conv import ConvBnRelu


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
    ):
        """bottleneck block in resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 2 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
        """
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(
                input_channel,
                block_base_channel,
                1,
                stride=stride,
                conv_last=True,
            ),
            ConvBnRelu(
                block_base_channel,
                block_base_channel,
                3,
                padding=1,
                conv_last=True,
            ),
            ConvBnRelu(
                block_base_channel,
                block_base_channel * 2,
                1,
                conv_last=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Hourglass2d(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_nested: int,
        num_resblock: int = 1,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            *[
                ResBlock(
                    hidden_dim,
                    hidden_dim,
                    block=BottleNeck(hidden_dim, hidden_dim // 2),
                )
                for _ in range(num_resblock if num_nested > 1 else num_resblock * 2)
            ]
        )
        self.block = (
            Hourglass2d(hidden_dim, num_nested - 1) if num_nested > 1 else nn.Identity()
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2, 2),
            *[
                ResBlock(
                    hidden_dim,
                    hidden_dim,
                    block=BottleNeck(hidden_dim, hidden_dim // 2),
                )
                for _ in range(num_resblock)
            ],
        )

        self.shortcut = nn.Sequential(
            *[
                ResBlock(
                    hidden_dim,
                    hidden_dim,
                    block=BottleNeck(hidden_dim, hidden_dim // 2),
                )
                for _ in range(num_resblock)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.downsample(x)
        y = self.block(y)
        y = self.upsample(y)
        y = F.interpolate(y, size=x.shape[-2:])

        shortcut = self.shortcut(x)

        return y + shortcut


class AuxiliaryHead(nn.Module):
    def __init__(
        self,
        input_channel: int = 256,
        output_size: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, input_channel, 1)
        self.output = nn.Conv2d(input_channel, output_size, 1)

        self.conv_neck = nn.Conv2d(input_channel, input_channel, 1)
        self.output_neck = nn.Conv2d(output_size, input_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        output_neck: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        y1 = self.conv(x)
        y = self.output(y1)

        y1 = self.conv_neck(y1)
        y1 = y1 + self.output_neck(y)

        if output_neck:
            return y, y1

        return y
