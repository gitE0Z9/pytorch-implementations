import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.network import ConvBnRelu


class ConvBlock(nn.Module):
    def __init__(self, input_channel: int, block_base_channel: int):
        """convolution block in resnet
        3 -> 3
        input_channel -> block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
        """
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(input_channel, block_base_channel, 3, padding=1),
            ConvBnRelu(
                block_base_channel,
                block_base_channel,
                3,
                padding=1,
                activation=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BottleNeck(nn.Module):
    def __init__(self, input_channel: int, block_base_channel: int):
        """bottleneck block in resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
        """
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
        """residual block in resnet
        skip connection has kernel size 1 and input_channel -> output_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            output_channel (int): output channel size
            block (ConvBlock | BottleNeck): block class
        """
        super(ResBlock, self).__init__()

        self.block = block(
            input_channel=input_channel,
            block_base_channel=block_base_channel,
        )

        self.downsample = (
            nn.Identity()
            if input_channel == output_channel
            else ConvBnRelu(input_channel, output_channel, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.downsample(x)
        return F.relu(y, inplace=True)
