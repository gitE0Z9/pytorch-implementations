import torch
from torch import nn

from torchlake.common.models import ConvBNReLU

from ..resnet.network import ConvBlock


class DropoutConvBlock(ConvBlock):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
        dropout_prob: float = 0.5,
    ):
        """convolution block in resnet
        3 -> 3
        input_channel -> block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            dropout_prob (float, Defaults 0.5): enable dropout layer between layers. Defaults to 0.5.
        """
        super(DropoutConvBlock, self).__init__(
            input_channel,
            block_base_channel,
            stride,
            pre_activation,
        )
        self.block.insert(1, nn.Dropout(dropout_prob))


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
        widening_factor: int = 1,
    ):
        """bottleneck block in resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            widening_factor (int, Defaults 1): width muliplier k in paper [1605.07146v4]. Defaults to 1.
        """
        super(BottleNeck, self).__init__()
        _block_base_channel = int(widening_factor * block_base_channel)
        self.block = nn.Sequential(
            ConvBNReLU(
                input_channel,
                _block_base_channel,
                1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                _block_base_channel,
                _block_base_channel,
                3,
                padding=1,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                _block_base_channel,
                block_base_channel * 4,
                1,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
