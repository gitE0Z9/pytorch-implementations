from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..inception.model import GoogLeNet
from ..inception.network import AuxiliaryClassifier
from .network import InceptionBlockV2


class InceptionBN(GoogLeNet):

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            # separable conv
            # paper states incorrect depth
            Conv2dNormActivation(input_channel, 64, 1),
            Conv2dNormActivation(64, 64, 7, stride=2, groups=8),
            nn.MaxPool2d(3, stride=2, padding=1),
            Conv2dNormActivation(64, 64, 1),
            Conv2dNormActivation(64, 192, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            InceptionBlockV2(192, (64, (64, 64), (64, 96, 96), 32)),
            InceptionBlockV2(256, (64, (64, 96), (64, 96, 96), 64)),
            InceptionBlockV2(
                320,
                (0, (128, 160), (64, 96, 96), 0),
                stride=2,
                pooling_type="max",
            ),
            InceptionBlockV2(576, (224, (64, 96), (96, 128, 128), 128)),
            # aux clf here
            InceptionBlockV2(576, (192, (96, 128), (96, 128, 128), 128)),
            InceptionBlockV2(576, (160, (128, 160), (128, 160, 160), 128)),
            # paper states incorrect number
            InceptionBlockV2(608, (96, (128, 192), (160, 192, 192), 128)),
            # aux clf here
            # paper states incorrect number
            InceptionBlockV2(
                608,
                (0, (128, 192), (192, 256, 256), 0),
                stride=2,
                pooling_type="max",
            ),
            # paper states incorrect number
            InceptionBlockV2(1056, (352, (192, 320), (160, 224, 224), 128)),
            InceptionBlockV2(
                1024,
                (352, (192, 320), (192, 224, 224), 128),
                pooling_type="max",
            ),
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                AuxiliaryClassifier(
                    576,
                    self.output_size,
                    hidden_dims=(128, 1024),
                    dropout_prob=self.aux_dropout_prob,
                    enable_bn=True,
                ),
                AuxiliaryClassifier(
                    608,
                    self.output_size,
                    hidden_dims=(128, 1024),
                    dropout_prob=self.aux_dropout_prob,
                    enable_bn=True,
                ),
            ]
        )
