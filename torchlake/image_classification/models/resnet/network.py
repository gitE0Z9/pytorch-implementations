from typing import Literal

import torch
from torch import nn
from torchlake.common.network import ConvBnRelu


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
        version: Literal["A"] | Literal["B"] | Literal["D"] = "A",
    ):
        """convolution block in resnet
        3 -> 3
        input_channel -> block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            version (Literal["A"] | Literal["B"] | Literal["D"], Defaults "A"): for compat.
        """
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(
                input_channel,
                block_base_channel,
                3,
                padding=1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBnRelu(
                block_base_channel,
                block_base_channel,
                3,
                padding=1,
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
        version: Literal["A"] | Literal["B"] | Literal["D"] = "A",
    ):
        """bottleneck block in resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            version (Literal["A"] | Literal["B"] | Literal["D"], Defaults "A"): A is resnet50 in paper[1512.03385], B, D is resnet-B, resnet-D in paper[1812.01187v2]. Defaults to "A".
        """
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(
                input_channel,
                block_base_channel,
                1,
                stride=stride if version == "A" else 1,
                conv_last=pre_activation,
            ),
            ConvBnRelu(
                block_base_channel,
                block_base_channel,
                3,
                padding=1,
                stride=stride if version in ["B", "D"] else 1,
                conv_last=pre_activation,
            ),
            ConvBnRelu(
                block_base_channel,
                block_base_channel * 4,
                1,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
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
        stride: int = 1,
        pre_activation: bool = False,
        version: Literal["A"] | Literal["B"] | Literal["D"] = "A",
    ):
        """residual block in resnet
        skip connection has kernel size 1 and input_channel -> output_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            output_channel (int): output channel size
            block (ConvBlock | BottleNeck): block class
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
            version (Literal["A"] | Literal["B"] | Literal["D"], Defaults "A"): A is resnet50 in paper[1512.03385], B, D is resnet-B, resnet-D in paper[1812.01187v2]. Defaults to "A".
        """
        super(ResBlock, self).__init__()
        self.pre_activation = pre_activation
        self.version = version

        # only apply version A, B, C, D to ResNet itself
        kwargs = dict(
            input_channel=input_channel,
            block_base_channel=block_base_channel,
            stride=stride,
            pre_activation=pre_activation,
        )
        if isinstance(block, ConvBlock) or isinstance(block, BottleNeck):
            kwargs["version"] = version
        self.block = block(**kwargs)

        self.downsample = self.build_shortcut(
            input_channel,
            output_channel,
            stride,
            version if version in ["A", "D"] else "A",
        )

        self.head = nn.Identity() if pre_activation else nn.ReLU(True)

    def build_shortcut(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        version: Literal["A"] | Literal["D"] = "A",
    ) -> nn.Module:
        """build shortcut

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            version (Literal["A"] | Literal["D"], Defaults "A"): A is resnet50 in paper[1512.03385], D is resnet-D in paper[1812.01187v2], Defaults to "A".
        """

        if input_channel == output_channel and stride == 1:
            return nn.Identity()

        layer = nn.Sequential(
            ConvBnRelu(
                input_channel,
                output_channel,
                1,
                stride=stride if version == "A" else 1,
                activation=None,
            )
        )

        if version == "D" and stride > 1:
            # use kenel size 3 for addition, since kernel size 2 not work
            layer.insert(0, nn.AvgPool2d(3, stride, padding=1))

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.downsample(x)

        return self.head(y)
