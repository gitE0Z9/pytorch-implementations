import torch
from torch import nn
from torchlake.common.models import ConvBNReLU


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """convolution block in resnet
        3 -> 3
        input_channel -> 2 * block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
        """
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBNReLU(
                input_channel,
                block_base_channel * 2,
                3,
                padding=1,
                stride=stride,
                group=32,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                block_base_channel * 2,
                block_base_channel,
                3,
                padding=1,
                group=32,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnext
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 2 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
        """
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            ConvBNReLU(
                input_channel,
                block_base_channel,
                1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                block_base_channel,
                block_base_channel,
                3,
                padding=1,
                group=32,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                block_base_channel,
                block_base_channel * 2,
                1,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
