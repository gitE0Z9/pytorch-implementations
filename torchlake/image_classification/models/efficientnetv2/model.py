from itertools import pairwise
from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..efficientnet.model import EfficientNet
from ..mobilenetv3.network import InvertedResidualBlockV3
from .network import InvertedResidualBlock


class EfficientNetV2(EfficientNet):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        enable_se: bool = True,
        dropout_prob: float = 0.2,
        key: Literal["s", "m", "l"] = "s",  # TODO: find xl
    ):
        """EfficientNet v2 in paper [2104.00298v3]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            enable_se (bool, optional): enable squeeze and excitation. Defaults to True.
            dropout_prob (float, optional): dropout prob, from 0.2 for S up to 0.5 for XL. Defaults to 0.2.
            key (Literal["s", "m", "l"], optional): key of configs. Defaults to "s".
        """
        self.enable_se = enable_se
        self.dropout_prob = dropout_prob
        self.key = key
        super().__init__(input_channel, output_size)

    @property
    def config(self) -> list[list[int]]:
        # channel, kernel, stride, expansion_ratio, block number
        return {
            "s": [
                [24, 3, 2],
                [24, 3, 1, 1, 2],
                [48, 3, 2, 4, 4],
                [64, 3, 2, 4, 4],  # end of fused mbconv
                [128, 3, 2, 4, 6],
                [160, 3, 1, 6, 9],
                [256, 3, 2, 6, 15],
            ],
            "m": [
                [24, 3, 2],
                [24, 3, 1, 1, 3],
                [48, 3, 2, 4, 5],
                [80, 3, 2, 4, 5],  # end of fused mbconv
                [160, 3, 2, 4, 7],
                [176, 3, 1, 6, 14],
                [304, 3, 2, 6, 18],
                [512, 3, 1, 6, 5],
            ],
            "l": [
                [32, 3, 2],
                [32, 3, 1, 1, 4],
                [64, 3, 2, 4, 7],
                [96, 3, 2, 4, 7],  # end of fused mbconv
                [192, 3, 2, 4, 10],
                [224, 3, 1, 6, 19],
                [384, 3, 2, 6, 25],
                [640, 3, 1, 6, 7],
            ],
        }[self.key]

    def build_blocks(self):
        blocks = []
        for stage_idx, (prev_layer, cur_layer) in enumerate(pairwise(self.config)):
            in_c = prev_layer[0]
            out_c, k, s, er, n = cur_layer

            block_class = (
                InvertedResidualBlock if stage_idx < 3 else InvertedResidualBlockV3
            )

            blocks.extend(
                [
                    block_class(
                        in_c if i == 0 else out_c,
                        out_c,
                        kernel=k,
                        stride=s if i == 0 else 1,
                        expansion_size=int((in_c if i == 0 else out_c) * er),
                        enable_se=self.enable_se,
                    )
                    for i in range(n)
                ]
            )

        self.blocks = nn.Sequential(
            *blocks,
            Conv2dNormActivation(
                out_c,
                1280,
                1,
                activation_layer=nn.SiLU,
                inplace=False,
            ),
        )
