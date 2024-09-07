import torch
from torch import nn
from torchlake.common.models import ConvBnRelu


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
        input_channel -> block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
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
    ):
        """bottleneck block in resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
        """
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            ConvBnRelu(
                input_channel,
                block_base_channel,
                1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBnRelu(
                block_base_channel,
                block_base_channel,
                3,
                padding=1,
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


class BottleNeckB(BottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnet-B in paper[1812.01187v2]
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before convolution layer in paper[1603.05027v3]
        """
        super(BottleNeckB, self).__init__(
            input_channel,
            block_base_channel,
            stride,
            pre_activation,
        )
        self.block[0].conv.stride = (1, 1)
        self.block[1].conv.stride = (stride, stride)


class BottleNeckD(BottleNeckB): ...


class ResBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        output_channel: int,
        block: ConvBlock | BottleNeck,
        stride: int = 1,
        pre_activation: bool = False,
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
        """
        super(ResBlock, self).__init__()
        self.pre_activation = pre_activation

        self.block = block(
            input_channel=input_channel,
            block_base_channel=block_base_channel,
            stride=stride,
            pre_activation=pre_activation,
        )

        self.downsample = self.build_shortcut(input_channel, output_channel, stride)

        self.head = nn.Identity() if pre_activation else nn.ReLU(True)

    def build_shortcut(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
    ) -> nn.Module:
        """build shortcut

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
        """

        if input_channel == output_channel and stride == 1:
            return nn.Identity()

        layer = nn.Sequential(
            ConvBnRelu(
                input_channel,
                output_channel,
                1,
                stride=stride,
                activation=None,
            )
        )

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.downsample(x)

        return self.head(y)


class ResBlockD(ResBlock):

    def build_shortcut(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
    ) -> nn.Module:
        """build shortcut

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
        """
        layer = super().build_shortcut(input_channel, output_channel, stride)

        if not isinstance(layer, nn.Identity):
            layer[0].conv.stride = (1, 1)

        if stride > 1:
            # use kenel size 3 for addition, since kernel size 2 not work
            layer.insert(0, nn.AvgPool2d(3, stride, padding=1))

        return layer
