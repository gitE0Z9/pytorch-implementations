from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock
from torchvision.ops import Conv2dNormActivation

from .network import darknet_bottleneck, darknet_block


class DarkNet53(ModelBase):

    @property
    def feature_dim(self) -> int:
        return 1024

    @property
    def config(self) -> list[list[list[int]]]:
        return [
            # channel, stride, block number, block class
            # stage 0
            [32, 2, 1, darknet_bottleneck],
            # stage 1
            [64, 2, 1, darknet_bottleneck],
            [64, 1, 1, darknet_block],
            # stage 2
            [128, 2, 1, darknet_bottleneck],
            [128, 1, 7, darknet_block],
            # stage 3
            [256, 2, 1, darknet_bottleneck],
            [256, 1, 7, darknet_block],
            # stage 4
            [512, 2, 1, darknet_bottleneck],
            [512, 1, 3, darknet_block],
        ]

    def build_foot(self, input_channel: int):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                32,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
        )

    def build_blocks(self):
        blocks = []
        for channel, stride, num_block, block in self.config:
            for _ in range(num_block):
                in_c = channel if block == darknet_bottleneck else channel * 2
                blocks.append(
                    ResBlock(
                        in_c,
                        channel * 2,
                        block(in_c, channel),
                        stride=stride,
                        activation=None,
                    ),
                )

        self.blocks = nn.Sequential(*blocks)
