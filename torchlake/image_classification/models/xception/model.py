from itertools import pairwise

from torch import nn
from torchlake.common.models import DepthwiseSeparableConv2d, ResBlock
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation


class Xception(ModelBase):

    @property
    def feature_dim(self) -> int:
        return 2048

    def build_foot(self, input_channel):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 32, 3, stride=2),
            Conv2dNormActivation(32, 64, 3),
            *[
                ResBlock(
                    in_c,
                    out_c,
                    nn.Sequential(
                        nn.Identity() if i == 0 else nn.ReLU(True),
                        DepthwiseSeparableConv2d(in_c, out_c),
                        DepthwiseSeparableConv2d(
                            out_c,
                            out_c,
                            activations=(None, None),
                        ),
                        nn.MaxPool2d(3, 2, padding=1),
                    ),
                    stride=2,
                    activation=None,
                )
                for i, (in_c, out_c) in enumerate(pairwise([64, 128, 256, 728]))
            ]
        )

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                ResBlock(
                    728,
                    728,
                    nn.Sequential(
                        nn.ReLU(True),
                        DepthwiseSeparableConv2d(728, 728),
                        DepthwiseSeparableConv2d(728, 728),
                        DepthwiseSeparableConv2d(
                            728,
                            728,
                            activations=(None, None),
                        ),
                    ),
                    activation=None,
                )
            ]
            * 8
        )

    def build_neck(self):
        self.neck = nn.Sequential(
            ResBlock(
                728,
                1024,
                nn.Sequential(
                    nn.ReLU(True),
                    DepthwiseSeparableConv2d(728, 728),
                    DepthwiseSeparableConv2d(728, 1024),
                    nn.MaxPool2d(3, 2, padding=1),
                ),
                stride=2,
                activation=None,
            ),
            DepthwiseSeparableConv2d(1024, 1536),
            DepthwiseSeparableConv2d(1536, 2048),
        )
