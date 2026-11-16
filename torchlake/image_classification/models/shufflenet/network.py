import torch
from torch import nn
from torchlake.common.models import ChannelShuffle
from torchvision.ops import Conv2dNormActivation


class BottleNeck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        groups: int = 1,
        reduction_ratio: int = 4,
        group_first_pointwise: bool = True,
    ):
        """Bottleneck in paper [1707.01083v2]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of depthwise convolution. Defaults to 1.
            groups (int, optional): number of groups of channels. Defaults to 1.
            reduction_ratio (int, optional): ratio to compress channel in bottleneck of resnet. Defaults to 4.
            group_first_pointwise (bool, optional): use group convolution in the first pointwise layer, paper disables it for the first block of stage 2 since its input channel is small. Defaults to True.
        """
        super().__init__()
        compressed_channel = output_channel // reduction_ratio
        if stride > 1:
            output_channel -= input_channel
        first_groups = groups if group_first_pointwise else 1

        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                compressed_channel,
                1,
                groups=first_groups,
            ),
            # shuffle
            Conv2dNormActivation(
                compressed_channel,
                compressed_channel,
                3,
                stride=stride,
                groups=compressed_channel,
                activation_layer=None,
            ),
            Conv2dNormActivation(
                compressed_channel,
                output_channel,
                1,
                groups=groups,
                activation_layer=None,
            ),
        )
        if first_groups > 1:
            self.layers.insert(1, ChannelShuffle(first_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResBlock(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        groups: int = 1,
        group_first_pointwise: bool = True,
    ):
        """residual block of shufflenet

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of depthwise convolution and identity branch. Defaults to 1.
            groups (int, optional): number of groups of channels. Defaults to 1.
            group_first_pointwise (bool, optional): use group convolution in the first pointwise layer of bottleneck. Defaults to True.
        """
        super().__init__()
        self.stride = stride

        self.block = BottleNeck(
            input_channel,
            output_channel,
            stride,
            groups,
            group_first_pointwise=group_first_pointwise,
        )
        self.identity_branch = (
            nn.AvgPool2d(3, 2, padding=1) if stride > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            y = self.identity_branch(x) + self.block(x)
        elif self.stride > 1:
            y = torch.cat([self.identity_branch(x), self.block(x)], 1)
        else:
            raise ValueError("Stride {} should be at least 1.".format(self.stride))

        return y.relu_()
