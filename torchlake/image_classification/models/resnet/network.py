import torch
from torch import nn
from torchlake.common.models import ConvBNReLU, ResBlock


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """convolution block in resnet
        kernel: 3x3 -> 3x3
        channel: input_channel -> output_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(
                input_channel,
                output_channel,
                3,
                padding=1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                output_channel,
                output_channel,
                3,
                padding=1,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnet
        kernel: 1x1 -> 3x3 -> 1x1
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 4.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(
                input_channel,
                hidden_dim,
                1,
                stride=stride,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                conv_last=pre_activation,
            ),
            ConvBNReLU(
                hidden_dim,
                hidden_dim * expansion_factor,
                1,
                activation=nn.ReLU(True) if pre_activation else None,
                conv_last=pre_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BottleNeckB(BottleNeck):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnet-B in paper[1812.01187v2]
        kernel: 1x1 -> 3x3 -> 1x1
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 4.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        self.layers[0].conv.stride = (1, 1)
        self.layers[1].conv.stride = (stride, stride)


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
        super().__init__()
        self.pre_activation = pre_activation

        self.block = block(
            input_channel=input_channel,
            block_base_channel=block_base_channel,
            stride=stride,
            pre_activation=pre_activation,
        )
        self.shortcut = self.build_shortcut(input_channel, output_channel, stride)
        self.activation = nn.Identity() if pre_activation else nn.ReLU(True)

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
            ConvBNReLU(
                input_channel,
                output_channel,
                1,
                stride=stride,
                activation=None,
                conv_last=self.pre_activation,
            )
        )

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x) + self.shortcut(x)

        return self.activation(y)


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

        if isinstance(layer, nn.Sequential):
            layer[0].conv.stride = (1, 1)
        # compatibility
        elif isinstance(layer, nn.Identity) and stride > 1:
            layer = nn.Sequential(layer)

        if stride > 1:
            # paper used that kernel size 2
            # but i found it does not match the size for addition so turn to kenel size 3
            layer.insert(0, nn.AvgPool2d(3, stride, padding=1))

        return layer
