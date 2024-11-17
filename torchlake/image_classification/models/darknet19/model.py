from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from .network import BottleNeck


class DarkNet19(ModelBase):
    @property
    def feature_dim(self) -> int:
        return 1024

    @property
    def config(self) -> list[list[list[int]]]:
        return [
            # channel, block number
            [128, 3],
            [256, 3],
            [512, 5],
            [1024, 5],
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
            nn.MaxPool2d(2, 2),
            Conv2dNormActivation(
                32,
                64,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.MaxPool2d(2, 2),
        )

    def build_blocks(self):
        blocks = []
        for stage_idx, stage in enumerate(self.config):
            c, n = stage
            blocks.append(BottleNeck(c, n))

            if stage_idx != len(self.config) - 1:
                blocks.append(nn.MaxPool2d(2, 2))

        self.blocks = nn.Sequential(*blocks)
