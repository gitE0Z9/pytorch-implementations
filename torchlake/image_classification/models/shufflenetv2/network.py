import torch
from torch import nn

from torchvision.ops import Conv2dNormActivation
from torchlake.common.models import ChannelShuffle


class BottleNeck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        reduction_ratio: int = 1,
    ):
        """Bottleneck in paper [1707.01083v2]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of depthwise convolution. Defaults to 1.
            reduction_ratio (int, optional): ratio to compress channel in bottleneck of resnet. Defaults to 1.
        """
        super(BottleNeck, self).__init__()
        compressed_channel = output_channel // reduction_ratio

        self.block = nn.Sequential(
            Conv2dNormActivation(input_channel, compressed_channel, 1),
            Conv2dNormActivation(
                compressed_channel,
                compressed_channel,
                3,
                stride=stride,
                groups=compressed_channel,
                activation_layer=None,
            ),
            Conv2dNormActivation(compressed_channel, output_channel, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        groups: int = 1,
    ):
        """_summary_

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of depthwise convolution and identity branch. Defaults to 1.
            groups (int, optional): number of groups of channel shuffle. Defaults to 1.
        """
        super(ResBlock, self).__init__()
        self.stride = stride

        _input_channel = input_channel if stride > 1 else input_channel // 2
        _output_channel = output_channel // 2

        self.block = BottleNeck(_input_channel, _output_channel, stride)
        self.identity_branch = (
            nn.Sequential(
                Conv2dNormActivation(
                    _input_channel,
                    _input_channel,
                    3,
                    stride=2,
                    groups=_input_channel,
                    activation_layer=None,
                ),
                Conv2dNormActivation(_input_channel, _output_channel, 1),
            )
            if stride > 1
            else nn.Identity()
        )
        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, 1)
            y = torch.cat([self.identity_branch(x1), self.block(x2)], 1)
        elif self.stride > 1:
            y = torch.cat([self.identity_branch(x), self.block(x)], 1)
        else:
            raise ValueError("Stride {} should be at least 1.".format(self.stride))

        return self.channel_shuffle(y)
