from torchlake.common.mixins.network import SEMixin
from torchlake.common.models import SqueezeExcitation2d

from ..resnet.network import BottleNeck, ConvBlock
from ..resnext.network import BottleNeck as XBottleNeck
from ..resnext.network import ConvBlock as XConvBlock


class SEConvBlock(SEMixin, ConvBlock):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
        reduction_ratio: int = 16,
    ):
        """convolution block in se-resnet
        kernel: 3x3 -> 3x3 -> se
        channel: input_channel -> output_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            reduction_ratio (int, optional): reduction ratio of squeeze and excitation. Defaults to 16.
        """
        super().__init__(
            input_channel,
            output_channel,
            stride,
            pre_activation,
        )
        self.se = SqueezeExcitation2d(output_channel, reduction_ratio)


class SEBottleNeck(SEMixin, BottleNeck):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        pre_activation: bool = False,
        reduction_ratio: int = 16,
    ):
        """bottleneck block in se-resnet
        kernel: 1x1 -> 3x3 -> 1x1 -> se
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 4.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            reduction_ratio (int, optional): reduction ratio of squeeze and excitation. Defaults to 16.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        self.se = SqueezeExcitation2d(hidden_dim * expansion_factor, reduction_ratio)


class SEXConvBlock(SEMixin, XConvBlock):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        group: int = 32,
        expansion_factor: int = 2,
        pre_activation: bool = False,
        reduction_ratio: int = 16,
    ):
        """convolution block in se-resnext
        kernel: 3x3 -> 3x3 -> se
        channel: input_channel -> expansion_factor * output_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            group (int, optional): group of middle layer. Defaults to 32.
            expansion_factor (int, optional): hidden dimension = expansion factor x output channel size. Defaults to 2.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            reduction_ratio (int, optional): reduction ratio of squeeze and excitation. Defaults to 16.
        """
        super().__init__(
            input_channel,
            output_channel,
            stride,
            group,
            expansion_factor,
            pre_activation,
        )
        self.se = SqueezeExcitation2d(output_channel, reduction_ratio)


class SEXBottleNeck(SEMixin, XBottleNeck):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        group: int = 32,
        expansion_factor: int = 2,
        pre_activation: bool = False,
        reduction_ratio: int = 16,
    ):
        """bottleneck block in se-resnext
        kernel: 1x1 -> 3x3 -> 1x1 -> se
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            group (int, optional): group of middle layer. Defaults to 32.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 2.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            reduction_ratio (int, optional): reduction ratio of squeeze and excitation. Defaults to 16.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            group,
            expansion_factor,
            pre_activation,
        )
        self.se = SqueezeExcitation2d(hidden_dim * expansion_factor, reduction_ratio)
