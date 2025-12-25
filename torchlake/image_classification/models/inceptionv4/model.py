from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock

from ..inceptionv3.network import InceptionBlockV3
from .network import ActivationScaling


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

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(1, nn.Dropout(p=self.dropout_prob))


class InceptionResNetV1(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.8,
        activation_scale: float = 0.1,
    ):
        self.dropout_prob = dropout_prob
        self.activation_scale = activation_scale
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 1792

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 32, 3, stride=2, padding=0),
            Conv2dNormActivation(32, 32, 3, padding=0),
            Conv2dNormActivation(32, 64, 3),
            nn.MaxPool2d(3, stride=2),
            Conv2dNormActivation(64, 80, 1),
            Conv2dNormActivation(80, 192, 3, padding=0),
            Conv2dNormActivation(192, 256, 3, padding=0),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()

        for _ in range(5):
            self.blocks.append(
                ResBlock(
                    256,
                    256,
                    block=nn.Sequential(
                        InceptionBlockV3(
                            256,
                            (32, (32, 32), (32, 32, 32), 0),
                            kernels=(1, (1, 3), (1, 3, 3)),
                            pooling_type=None,
                        ),
                        Conv2dNormActivation(96, 256, 1, activation_layer=None),
                        (
                            ActivationScaling(scale=self.activation_scale)
                            if self.activation_scale > 0
                            else nn.Identity()
                        ),
                    ),
                )
            )

        # reduction A
        self.blocks.append(
            InceptionBlockV3(
                256,
                (384, (192, 192, 256), 0),
                kernels=(3, (1, 3, 3)),
                stride=2,
                pooling_type="max",
            )
        )

        for _ in range(10):
            self.blocks.append(
                ResBlock(
                    896,
                    896,
                    block=nn.Sequential(
                        InceptionBlockV3(
                            896,
                            (128, (128, (128, 128)), 0),
                            kernels=(1, (1, (7, False))),
                            pooling_type=None,
                        ),
                        Conv2dNormActivation(256, 896, 1, activation_layer=None),
                        (
                            ActivationScaling(scale=self.activation_scale)
                            if self.activation_scale > 0
                            else nn.Identity()
                        ),
                    ),
                )
            )

        # reduction B
        self.blocks.append(
            InceptionBlockV3(
                896,
                ((256, 384), (256, 256), (256, 256, 256), 0),
                kernels=((1, 3), (1, 3), (1, 3, 3)),
                stride=2,
                pooling_type="max",
            )
        )

        for _ in range(5):
            self.blocks.append(
                ResBlock(
                    1792,
                    1792,
                    block=nn.Sequential(
                        InceptionBlockV3(
                            1792,
                            (192, (192, (192, 192)), 0),
                            kernels=(1, (1, (3, False))),
                            pooling_type=None,
                        ),
                        Conv2dNormActivation(384, 1792, 1, activation_layer=None),
                        (
                            ActivationScaling(scale=self.activation_scale)
                            if self.activation_scale > 0
                            else nn.Identity()
                        ),
                    ),
                )
            )

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(1, nn.Dropout(p=self.dropout_prob))


class InceptionResNetV2(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        dropout_prob: float = 0.8,
        activation_scale: float = 0.1,
    ):
        self.dropout_prob = dropout_prob
        self.activation_scale = activation_scale
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 2048

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

        for _ in range(5):
            self.blocks.append(
                ResBlock(
                    384,
                    384,
                    block=nn.Sequential(
                        InceptionBlockV3(
                            384,
                            (32, (32, 32), (32, 48, 64), 0),
                            kernels=(1, (1, 3), (1, 3, 3)),
                            pooling_type=None,
                        ),
                        Conv2dNormActivation(128, 384, 1, activation_layer=None),
                        (
                            ActivationScaling(scale=self.activation_scale)
                            if self.activation_scale > 0
                            else nn.Identity()
                        ),
                    ),
                )
            )

        # reduction A
        self.blocks.append(
            InceptionBlockV3(
                384,
                (384, (256, 256, 384), 0),
                kernels=(3, (1, 3, 3)),
                stride=2,
                pooling_type="max",
            )
        )

        for i in range(10):
            self.blocks.append(
                ResBlock(
                    1154 if i > 0 else 1152,
                    1154,
                    block=nn.Sequential(
                        InceptionBlockV3(
                            1154 if i > 0 else 1152,
                            (192, (128, (160, 192)), 0),
                            kernels=(1, (1, (7, False))),
                            pooling_type=None,
                        ),
                        Conv2dNormActivation(384, 1154, 1, activation_layer=None),
                        (
                            ActivationScaling(scale=self.activation_scale)
                            if self.activation_scale > 0
                            else nn.Identity()
                        ),
                    ),
                )
            )

        # reduction B
        self.blocks.append(
            InceptionBlockV3(
                1154,
                ((256, 384), (256, 288), (256, 288, 320), 0),
                kernels=((1, 3), (1, 3), (1, 3, 3)),
                stride=2,
                pooling_type="max",
            )
        )

        for i in range(5):
            self.blocks.append(
                ResBlock(
                    2048 if i > 0 else 2146,
                    2048,
                    block=nn.Sequential(
                        InceptionBlockV3(
                            2048 if i > 0 else 2146,
                            (192, (192, (224, 256)), 0),
                            kernels=(1, (1, (3, False))),
                            pooling_type=None,
                        ),
                        Conv2dNormActivation(448, 2048, 1, activation_layer=None),
                        (
                            ActivationScaling(scale=self.activation_scale)
                            if self.activation_scale > 0
                            else nn.Identity()
                        ),
                    ),
                )
            )

    def build_head(self, output_size, **kwargs):
        super().build_head(output_size, **kwargs)
        self.head.insert(1, nn.Dropout(p=self.dropout_prob))
