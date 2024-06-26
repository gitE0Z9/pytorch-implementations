import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.network import ConvBnRelu


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        pre_activation: bool = False,
    ):
        """convolution block in resnet
        3 -> 3
        input_channel -> block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            pre_activation (bool, Defaults False):
        """
        super(ConvBlock, self).__init__()
        self.pre_activation = pre_activation

        self.block = nn.Sequential(
            ConvBnRelu(input_channel, block_base_channel * 2, 3, padding=1, group=32),
            ConvBnRelu(
                block_base_channel * 2,
                block_base_channel,
                3,
                padding=1,
                group=32,
                activation=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_activation:
            x = F.relu(x, True)
        return self.block(x)


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnext
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            pre_activation (bool, Defaults False): activation before block
        """
        super(BottleNeck, self).__init__()
        self.pre_activation = pre_activation

        self.block = nn.Sequential(
            ConvBnRelu(input_channel, block_base_channel, 1),
            ConvBnRelu(block_base_channel, block_base_channel, 3, padding=1, group=32),
            ConvBnRelu(
                block_base_channel,
                block_base_channel * 2,
                1,
                activation=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_activation:
            x = F.relu(x, True)
        return self.block(x)
