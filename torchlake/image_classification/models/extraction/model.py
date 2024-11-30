from itertools import pairwise

from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation


class Extraction(ModelBase):

    @property
    def feature_dim(self) -> int:
        return 1024

    @property
    def config(self) -> list[list[list[int]]]:
        return [
            [
                [192],
                [128, 1],
                [256, 3],
                [256, 1],
                [512, 3],
            ],
            [
                [512],
                [256, 1],
                [512, 3],
                [256, 1],
                [512, 3],
                [256, 1],
                [512, 3],
                [256, 1],
                [512, 3],
                [512, 1],
                [1024, 3],
            ],
            [
                [1024],
                [512, 1],
                [1024, 3],
                [512, 1],
                [1024, 3],
            ],
        ]

    def build_foot(self, input_channel: int):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                64,
                7,
                stride=2,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.MaxPool2d(2, 2),
            Conv2dNormActivation(
                64,
                192,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.MaxPool2d(2, 2),
        )

    def build_blocks(self):
        blocks = []
        for stage_idx, stage in enumerate(self.config):
            for prev_block, next_block in pairwise(stage):
                in_c, out_c, kernel = prev_block[0], next_block[0], next_block[1]
                blocks.append(
                    Conv2dNormActivation(
                        in_c,
                        out_c,
                        kernel,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    )
                )

            if stage_idx != len(self.config) - 1:
                blocks.append(nn.MaxPool2d(2, 2))

        self.blocks = nn.Sequential(*blocks)

    # def build_head(self, output_size: int):
    #     self.head = nn.Sequential(
    #         Conv2dNormActivation(
    #             self.feature_dim,
    #             output_size,
    #             1,
    #             padding=1,
    #             norm_layer=None,
    #             activation_layer=lambda: nn.LeakyReLU(0.1),
    #             inplace=None,
    #         ),
    #         FlattenFeature(),
    #     )
