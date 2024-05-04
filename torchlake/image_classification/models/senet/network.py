from torchlake.common.mixins.network import SeMixin
from torchlake.common.network import SqueezeExcitation2d

from ..resnet.network import BottleNeck, ConvBlock
from ..resnext.network import BottleNeck as XBottleNeck
from ..resnext.network import ConvBlock as XConvBlock


class SeConvBlock(SeMixin, ConvBlock):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        pre_activation: bool = False,
    ):
        """convolution block in se-resnet
        3 -> 3
        input_channel -> block_base_channel -> block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            pre_activation (bool, Defaults False): put activation before transformation [1603.05027v3]
        """
        super(SeConvBlock, self).__init__(
            input_channel,
            block_base_channel,
            pre_activation,
        )
        self.se = SqueezeExcitation2d(block_base_channel, 16)


class SeBottleNeck(SeMixin, BottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        pre_activation: bool = False,
    ):
        """bottleneck block in se-resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            pre_activation (bool, Defaults False): put activation before transformation [1603.05027v3]
        """
        super(SeBottleNeck, self).__init__(
            input_channel,
            block_base_channel,
            pre_activation,
        )
        self.se = SqueezeExcitation2d(block_base_channel * 4, 16)


class SeXConvBlock(SeMixin, XConvBlock):
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
        super(SeXConvBlock, self).__init__(
            input_channel,
            block_base_channel,
            pre_activation,
        )
        self.pre_activation = pre_activation

        self.se = SqueezeExcitation2d(block_base_channel, 16)


class SeXBottleNeck(SeMixin, XBottleNeck):
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
        super(SeXBottleNeck, self).__init__(
            input_channel,
            block_base_channel,
            pre_activation,
        )

        self.se = SqueezeExcitation2d(block_base_channel * 2, 16)
