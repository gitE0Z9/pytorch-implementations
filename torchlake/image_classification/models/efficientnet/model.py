from itertools import pairwise

from torch import nn
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from ..mobilenetv2.network import InvertedResidualBlock

# params_dict = {
#       # (width_coefficient, depth_coefficient, resolution, dropout_rate)
#       'efficientnet-b0': (1.0, 1.0, 224, 0.2),
#       'efficientnet-b1': (1.0, 1.1, 240, 0.2),
#       'efficientnet-b2': (1.1, 1.2, 260, 0.3),
#       'efficientnet-b3': (1.2, 1.4, 300, 0.3),
#       'efficientnet-b4': (1.4, 1.8, 380, 0.4),
#       'efficientnet-b5': (1.6, 2.2, 456, 0.4),
#       'efficientnet-b6': (1.8, 2.6, 528, 0.5),
#       'efficientnet-b7': (2.0, 3.1, 600, 0.5),
#       'efficientnet-b8': (2.2, 3.6, 672, 0.5),
#       'efficientnet-l2': (4.3, 5.3, 800, 0.5),
#   }


class EfficientNet(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        depth_multiplier: float = 1.0,
        width_multiplier: float = 1.0,
        dropout_prob: float = 0.2,
    ):
        """EfficientNet B0 in paper [1905.11946v5]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            depth_multiplier (float, optional): alpha. Defaults to 1.2.
            width_multiplier (float, optional): beta. Defaults to 1.1.
            dropout_prob (float, optional): dropout prob, from 0.2 for B0 up to 0.5 for B7. Defaults to 0.2.
        """
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.dropout_prob = dropout_prob
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return int(1280 * self.width_multiplier)

    @property
    def config(self) -> list[list[int]]:
        return [
            # channel, kernel, stride, block number
            [16, 3, 1, 1],
            [24, 3, 1, 2],
            [40, 5, 2, 2],
            [80, 3, 2, 3],
            [112, 5, 2, 3],
            [192, 5, 1, 4],
            [320, 3, 2, 1],
        ]

    def build_foot(self, input_channel):
        self.foot = Conv2dNormActivation(
            input_channel,
            int(32 * self.width_multiplier),
            3,
            stride=2,
        )

    def build_blocks(self):
        cfg = [
            [
                int(c * self.width_multiplier),
                k,
                s,
                int(n * self.depth_multiplier),
            ]
            for c, k, s, n in self.config
        ]

        blocks = []
        out_c, k, s, n = cfg[0]
        blocks.extend(
            [
                InvertedResidualBlock(
                    int(32 * self.width_multiplier) if i == 0 else out_c,
                    out_c,
                    kernel=k,
                    stride=s if i == 0 else 1,
                )
                for i in range(n)
            ]
        )
        for prev_layer, cur_layer in pairwise(cfg):
            in_c = prev_layer[0]
            out_c, k, s, n = cur_layer

            blocks.extend(
                [
                    InvertedResidualBlock(
                        in_c if i == 0 else out_c,
                        out_c,
                        kernel=k,
                        stride=s if i == 0 else 1,
                    )
                    for i in range(n)
                ]
            )

        self.blocks = nn.Sequential(
            *blocks,
            Conv2dNormActivation(
                int(320 * self.width_multiplier),
                int(1280 * self.width_multiplier),
                1,
            ),
        )

    def build_neck(self):
        self.neck = nn.Dropout(self.dropout_prob)


# image size: 224, gamma: 1
def efficient_b0(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(input_channel, output_size)


# image size: 240, gamma: 1.07
def efficient_b1(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        depth_multiplier=1.1,
    )


# image size: 260, gamma: 1.16
def efficient_b2(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=1.1,
        depth_multiplier=1.2,
        dropout_prob=0.3,
    )


# image size: 300, gamma: 1.34
def efficient_b3(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=1.2,
        depth_multiplier=1.4,
        dropout_prob=0.3,
    )


# image size: 380, gamma: 1.696
def efficient_b4(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=1.4,
        depth_multiplier=1.8,
        dropout_prob=0.4,
    )


# image size: 456, gamma: 2.036
def efficient_b5(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=1.6,
        depth_multiplier=2.2,
        dropout_prob=0.4,
    )


# image size: 528, gamma: 2.357
def efficient_b6(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=1.8,
        depth_multiplier=2.6,
        dropout_prob=0.5,
    )


# image size: 600, gamma: 2.68
def efficient_b7(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=2.0,
        depth_multiplier=3.1,
        dropout_prob=0.5,
    )


# image size: 672, gamma: 3
def efficient_b8(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=2.2,
        depth_multiplier=3.6,
        dropout_prob=0.5,
    )


# image size: 800, gamma: 3.57
def efficient_l2(
    input_channel: int = 3,
    output_size: int = 1,
):

    return EfficientNet(
        input_channel,
        output_size,
        width_multiplier=4.3,
        depth_multiplier=5.3,
        dropout_prob=0.5,
    )
