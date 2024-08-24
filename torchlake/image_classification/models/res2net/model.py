from torch import nn
from torchlake.common.models import ResBlock

from ..resnet.model import ResNet
from .network import BottleNeck

# input, output, base?, num_repeat, block_type
CONFIGS = {
    50: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 4, BottleNeck],
        [512, 1024, 256, 6, BottleNeck],
        [1024, 2048, 512, 3, BottleNeck],
    ],
    101: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 4, BottleNeck],
        [512, 1024, 256, 23, BottleNeck],  # more block
        [1024, 2048, 512, 3, BottleNeck],
    ],
    152: [
        [64, 256, 64, 3, BottleNeck],
        [256, 512, 128, 8, BottleNeck],  # more block
        [512, 1024, 256, 36, BottleNeck],  # more block
        [1024, 2048, 512, 3, BottleNeck],
    ],
}


class Res2Net(ResNet):
    configs = CONFIGS

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: int = 50,
        split: int = 4,
        groups: int = 1,
    ):
        """Res2Net in paper [1904.01169v3]

        Args:
            input_channel (int, optional): input channel size. Defaults to 3.
            output_size (int, optional): output channel size. Defaults to 1.
            key (int, optional): key of configs. Defaults to 50.
            split (int, optional): s in paper, split input_channel into s * w channels. Defaults to 4.
            groups (int, optional): group of 3x3 filters. Defaults to 1.
        """
        self.split = split
        self.groups = groups
        super(Res2Net, self).__init__(
            input_channel,
            output_size,
            key=key,
        )

        delattr(self, "pre_activation")

    def build_blocks(self):
        """build blocks"""
        for block_index, (
            input_channel,
            output_channel,
            base_number,
            num_layer,
            layer_class,
        ) in enumerate(self.config):
            layers = nn.Sequential()

            for layer_index in range(num_layer):
                layer = layer_class(
                    input_channel if layer_index == 0 else output_channel,
                    base_number,
                    stride=2 if layer_index == 0 else 1,
                    split=self.split,
                    groups=self.groups,
                )
                layers.append(
                    ResBlock(
                        input_channel if layer_index == 0 else output_channel,
                        output_channel,
                        layer,
                        stride=2 if layer_index == 0 else 1,
                    )
                )

            setattr(self, f"block{block_index+1}", layers)
