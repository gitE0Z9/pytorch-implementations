import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

from .network import AuxiliaryClassifierV3, InceptionBlockV3


class InceptionV3(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.5,
        aux_dropout_prob: float = 0,
    ):
        self.dropout_prob = dropout_prob
        self.aux_dropout_prob = aux_dropout_prob
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 2048

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 32, 3, stride=2, padding=0),
            Conv2dNormActivation(32, 32, 3, padding=0),
            Conv2dNormActivation(32, 64, 3),
            nn.MaxPool2d(3, stride=2),
            Conv2dNormActivation(64, 80, 1),
            Conv2dNormActivation(80, 192, 3, padding=0),
            nn.MaxPool2d(3, stride=2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            InceptionBlockV3(
                192,
                (64, (48, 64), (64, 96, 96), 32),
                kernels=(
                    1,
                    (1, 5),
                    (1, 3, 3),
                ),
            ),
            InceptionBlockV3(
                256,
                (64, (48, 64), (64, 96, 96), 64),
                kernels=(
                    1,
                    (1, 5),
                    (1, 3, 3),
                ),
            ),
            InceptionBlockV3(
                288,
                (64, (48, 64), (64, 96, 96), 64),
                kernels=(
                    1,
                    (1, 5),
                    (1, 3, 3),
                ),
            ),
            InceptionBlockV3(
                288,
                (384, (64, 96, 96), 0),
                kernels=(3, (1, 3, 3)),
                stride=2,
                pooling_type="max",
            ),
            InceptionBlockV3(
                768,
                (192, (128, (128, 192)), (128, (128, 128), (128, 192)), 192),
                kernels=(1, (1, (7, False)), (1, (7, True), (7, True))),
            ),
            InceptionBlockV3(
                768,
                (192, (160, (160, 192)), (160, (160, 160), (160, 192)), 192),
                kernels=(1, (1, (7, False)), (1, (7, True), (7, True))),
            ),
            InceptionBlockV3(
                768,
                (192, (160, (160, 192)), (160, (160, 160), (160, 192)), 192),
                kernels=(1, (1, (7, False)), (1, (7, True), (7, True))),
            ),
            # 6e
            InceptionBlockV3(
                768,
                (192, (192, (192, 192)), (192, (192, 192), (192, 192)), 192),
                kernels=(1, (1, (7, False)), (1, (7, True), (7, True))),
            ),
            # aux clf here
            # 7a
            InceptionBlockV3(
                768,
                ((192, 320), (192, (192, 192), 192), 0),
                kernels=((1, 3), (1, (7, False), 3)),
                stride=2,
                pooling_type="max",
            ),
            InceptionBlockV3(
                1280,
                (
                    320,
                    (384, (384, 384)),
                    (448, 384, (384, 384)),
                    192,
                ),
                kernels=(
                    1,
                    (1, (3, False, None)),
                    (1, 3, (3, False, None)),
                ),
            ),
            InceptionBlockV3(
                2048,
                (
                    320,
                    (384, (384, 384)),
                    (448, 384, (384, 384)),
                    192,
                ),
                kernels=(
                    1,
                    (1, (3, False, None)),
                    (1, 3, (3, False, None)),
                ),
            ),
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                AuxiliaryClassifierV3(
                    768,
                    self.output_size,
                    hidden_dims=(128, 768),
                    kernel=5,
                    dropout_prob=self.aux_dropout_prob,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        y = self.foot(x)

        if self.training:
            outputs = []

            y = self.blocks[:8](y)
            outputs.append(self.neck[0](y))

            y = self.blocks[8:](y)
            outputs.append(self.head(y))

            return outputs
        else:
            y = self.blocks(y)
            return self.head(y)
