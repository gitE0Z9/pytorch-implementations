from ..resnet.model import ResNet
from .network import BottleNeck, ConvBlock

# input, output, base?, number_block, block_type
CONFIGS = {
    18: [
        [64, 64, 64, 2, ConvBlock],  # less block
        [64, 128, 128, 2, ConvBlock],  # less block
        [128, 256, 256, 2, ConvBlock],  # less block
        [256, 512, 512, 2, ConvBlock],  # less block
    ],
    34: [
        [64, 64, 64, 3, ConvBlock],  # narrower
        [64, 128, 128, 4, ConvBlock],  # narrower
        [128, 256, 256, 6, ConvBlock],  # narrower
        [256, 512, 512, 3, ConvBlock],  # narrower
    ],
    50: [
        [64, 256, 128, 3, BottleNeck],
        [256, 512, 256, 4, BottleNeck],
        [512, 1024, 512, 6, BottleNeck],
        [1024, 2048, 1024, 3, BottleNeck],
    ],
    101: [
        [64, 256, 128, 3, BottleNeck],
        [256, 512, 256, 4, BottleNeck],
        [512, 1024, 512, 23, BottleNeck],  # more block
        [1024, 2048, 1024, 3, BottleNeck],
    ],
    152: [
        [64, 256, 128, 3, BottleNeck],
        [256, 512, 256, 8, BottleNeck],  # more block
        [512, 1024, 512, 36, BottleNeck],  # more block
        [1024, 2048, 1024, 3, BottleNeck],
    ],
}


class ResNeXt(ResNet):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 50,
        pre_activation: bool = False,
        configs=CONFIGS,
    ):
        super(ResNeXt, self).__init__(
            input_channel,
            output_size,
            num_layer,
            pre_activation,
            configs,
        )
