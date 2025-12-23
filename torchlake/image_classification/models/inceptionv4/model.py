from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase

from ..inceptionv3.network import InceptionBlockV3


class InceptionV4(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.8,
    ):
        self.dropout_prob = dropout_prob
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 1536

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 32, 3, stride=2, padding=0),
            Conv2dNormActivation(32, 32, 3, padding=0),
            Conv2dNormActivation(32, 64, 3),
            InceptionBlockV3(64, (96, 0), kernels=(3,), pooling_type="max", stride=2),
            InceptionBlockV3(
                160,
                ((64, 96), (64, (64, 64), 96), 0),
                kernels=((1, 3), (1, (7, True), 3)),
                pooling_type=None,
            ),
            InceptionBlockV3(192, (192, 0), kernels=(3,), pooling_type="max", stride=2),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()

        for _ in range(4):
            self.blocks.append(
                InceptionBlockV3(
                    384,
                    (96, (64, 96), (64, 96, 96), 96),
                    kernels=(1, (1, 3), (1, 3, 3)),
                )
            )

        # reduction A
        self.blocks.append(
            InceptionBlockV3(
                384,
                (384, (192, 224, 256), 0),
                kernels=(3, (1, 3, 3)),
                stride=2,
                pooling_type="max",
            )
        )

        for _ in range(7):
            self.blocks.append(
                InceptionBlockV3(
                    1024,
                    (384, (192, (224, 256)), (192, (192, 224), (224, 256)), 128),
                    kernels=(1, (1, (7, False)), (1, (7, False), (7, False))),
                )
            )

        # reduction B
        self.blocks.append(
            InceptionBlockV3(
                1024,
                ((192, 192), (256, (256, 320), 320), 0),
                kernels=((1, 3), (1, (7, False), 3)),
                stride=2,
                pooling_type="max",
            )
        )

        for _ in range(3):
            self.blocks.append(
                InceptionBlockV3(
                    1536,
                    (256, (384, (256, 256)), (384, (448, 512), (256, 256)), 256),
                    kernels=(
                        1,
                        (1, (3, False, None)),
                        (1, (3, False), (3, True, None)),
                    ),
                )
            )
