from ..resnet.model import ResNet
from ..resnext.model import ResNeXt
from .network import SeBottleNeck, SeConvBlock, SeXBottleNeck, SeXConvBlock

CONFIGS = {
    18: [
        [64, 64, 64, 2, SeConvBlock],  # less block
        [64, 128, 128, 2, SeConvBlock],  # less block
        [128, 256, 256, 2, SeConvBlock],  # less block
        [256, 512, 512, 2, SeConvBlock],  # less block
    ],
    34: [
        [64, 64, 64, 3, SeConvBlock],  # narrower
        [64, 128, 128, 4, SeConvBlock],  # narrower
        [128, 256, 256, 6, SeConvBlock],  # narrower
        [256, 512, 512, 3, SeConvBlock],  # narrower
    ],
    50: [
        [64, 256, 64, 3, SeBottleNeck],
        [256, 512, 128, 4, SeBottleNeck],
        [512, 1024, 256, 6, SeBottleNeck],
        [1024, 2048, 512, 3, SeBottleNeck],
    ],
    101: [
        [64, 256, 64, 3, SeBottleNeck],
        [256, 512, 128, 4, SeBottleNeck],
        [512, 1024, 256, 23, SeBottleNeck],  # more block
        [1024, 2048, 512, 3, SeBottleNeck],
    ],
    152: [
        [64, 256, 64, 3, SeBottleNeck],
        [256, 512, 128, 8, SeBottleNeck],  # more block
        [512, 1024, 256, 36, SeBottleNeck],  # more block
        [1024, 2048, 512, 3, SeBottleNeck],
    ],
}


class SeResNet(ResNet):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 50,
        pre_activation: bool = False,
        configs=CONFIGS,
    ):
        super(SeResNet, self).__init__(
            input_channel,
            output_size,
            num_layer,
            pre_activation,
            configs,
        )


XCONFIGS = {
    18: [
        [64, 64, 64, 2, SeXConvBlock],  # less block
        [64, 128, 128, 2, SeXConvBlock],  # less block
        [128, 256, 256, 2, SeXConvBlock],  # less block
        [256, 512, 512, 2, SeXConvBlock],  # less block
    ],
    34: [
        [64, 64, 64, 3, SeXConvBlock],  # narrower
        [64, 128, 128, 4, SeXConvBlock],  # narrower
        [128, 256, 256, 6, SeXConvBlock],  # narrower
        [256, 512, 512, 3, SeXConvBlock],  # narrower
    ],
    50: [
        [64, 256, 128, 3, SeXBottleNeck],
        [256, 512, 256, 4, SeXBottleNeck],
        [512, 1024, 512, 6, SeXBottleNeck],
        [1024, 2048, 1024, 3, SeXBottleNeck],
    ],
    101: [
        [64, 256, 128, 3, SeXBottleNeck],
        [256, 512, 256, 4, SeXBottleNeck],
        [512, 1024, 512, 23, SeXBottleNeck],  # more block
        [1024, 2048, 1024, 3, SeXBottleNeck],
    ],
    152: [
        [64, 256, 128, 3, SeXBottleNeck],
        [256, 512, 256, 8, SeXBottleNeck],  # more block
        [512, 1024, 512, 36, SeXBottleNeck],  # more block
        [1024, 2048, 1024, 3, SeXBottleNeck],
    ],
}


class SeResNeXt(ResNeXt):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 50,
        pre_activation: bool = False,
        configs=XCONFIGS,
    ):
        super(SeResNeXt, self).__init__(
            input_channel,
            output_size,
            num_layer,
            pre_activation,
            configs,
        )
