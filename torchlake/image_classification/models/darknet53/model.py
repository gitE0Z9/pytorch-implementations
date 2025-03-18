from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock
from torchvision.ops import Conv2dNormActivation

from .network import DarkNetBlock


class DarkNet53(ModelBase):

    @property
    def feature_dim(self) -> int:
        return 1024

    @property
    def config(self) -> list[list[list[int]]]:
        return [
            # channel, block number, block class
            # stage 0
            [32, 1],
            # stage 1
            [64, 2],
            # stage 2
            [128, 8],
            # stage 3
            [256, 8],
            # stage 4
            [512, 4],
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
        for channel, num_block in self.config:
            blocks.append(
                Conv2dNormActivation(
                    channel,
                    channel * 2,
                    3,
                    stride=2,
                    activation_layer=lambda: nn.LeakyReLU(0.1),
                    inplace=None,
                ),
            )
            for _ in range(num_block):
                blocks.append(
                    ResBlock(
                        channel * 2,
                        channel * 2,
                        DarkNetBlock(channel * 2, channel),
                        activation=None,
                    ),
                )

        self.blocks = nn.Sequential(*blocks)
