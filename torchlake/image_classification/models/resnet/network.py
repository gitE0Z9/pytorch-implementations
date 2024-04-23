import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.network import ConvBnRelu


class ConvBlock(nn.Module):
    def __init__(self, input_channel: int, block_base_channel: int):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(input_channel, block_base_channel, 3, padding=1),
            ConvBnRelu(block_base_channel, block_base_channel, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BottleNeck(nn.Module):
    def __init__(self, input_channel: int, block_base_channel: int):
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(input_channel, block_base_channel, 1),
            ConvBnRelu(block_base_channel, block_base_channel, 3, padding=1),
            ConvBnRelu(
                block_base_channel,
                block_base_channel * 4,
                1,
                activation=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        output_channel: int,
        block: ConvBlock | BottleNeck,
    ):
        super(ResBlock, self).__init__()

        self.block = block(
            input_channel=input_channel,
            block_base_channel=block_base_channel,
        )

        self.downsample = (
            nn.Identity()
            if input_channel == output_channel
            else nn.Sequential(ConvBnRelu(input_channel, output_channel, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.downsample(x)
        return F.relu(y, inplace=True)
