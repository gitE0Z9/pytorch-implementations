from typing import Literal, NamedTuple

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.residual import ResBlock

from .network import FusedInvertedBottleneck, MobileMQA, UniversalInvertedBottleneck


class ConvArgs(NamedTuple):
    output_dim: int
    kernel: int
    stride: int


class FusedIBArgs(NamedTuple):
    output_dim: int
    kernel: int
    stride: int
    expansion_ratio: float


class UIBArgs(NamedTuple):
    output_dim: int
    start_dw_kernel: int
    middle_dw_kernel: int
    stride: int
    expansion_ratio: float


class MQAArgs(NamedTuple):
    num_head: int
    key_dim: int
    value_dim: int
    kv_stride: int


LAYER_ARGS = ConvArgs | FusedIBArgs | UIBArgs | MQAArgs

# official specs: tensorflow/models official/vision/modeling/backbones/mobilenet.py
CONV_SMALL = (
    ConvArgs(32, 3, 2),
    # paper: FusedIBArgs(32, 3, 2, 1)
    *[ConvArgs(32, 3, 2), ConvArgs(32, 1, 1)],
    # paper: FusedIBArgs(64, 3, 2, 3)
    *[ConvArgs(96, 3, 2), ConvArgs(64, 1, 1)],
    UIBArgs(96, 5, 5, 2, 3),
    *[UIBArgs(96, 0, 3, 1, 2)] * 4,
    UIBArgs(96, 3, 0, 1, 4),
    UIBArgs(128, 3, 3, 2, 6),
    UIBArgs(128, 5, 5, 1, 4),
    UIBArgs(128, 0, 5, 1, 4),
    UIBArgs(128, 0, 5, 1, 3),
    *[UIBArgs(128, 0, 3, 1, 4)] * 2,
)

CONV_MEDIUM = (
    ConvArgs(32, 3, 2),
    FusedIBArgs(48, 3, 2, 4),
    UIBArgs(80, 3, 5, 2, 4),
    UIBArgs(80, 3, 3, 1, 2),
    UIBArgs(160, 3, 5, 2, 6),
    *[UIBArgs(160, 3, 3, 1, 4)] * 2,
    UIBArgs(160, 3, 5, 1, 4),
    UIBArgs(160, 3, 3, 1, 4),
    UIBArgs(160, 3, 0, 1, 4),
    UIBArgs(160, 0, 0, 1, 2),
    UIBArgs(160, 3, 0, 1, 4),
    UIBArgs(256, 5, 5, 2, 6),
    UIBArgs(256, 5, 5, 1, 4),
    *[UIBArgs(256, 3, 5, 1, 4)] * 2,
    UIBArgs(256, 0, 0, 1, 4),
    UIBArgs(256, 3, 0, 1, 4),
    UIBArgs(256, 3, 5, 1, 2),
    UIBArgs(256, 5, 5, 1, 4),
    *[UIBArgs(256, 0, 0, 1, 4)] * 2,
    UIBArgs(256, 5, 0, 1, 2),
)

CONV_LARGE = (
    ConvArgs(24, 3, 2),
    FusedIBArgs(48, 3, 2, 4),
    UIBArgs(96, 3, 5, 2, 4),
    UIBArgs(96, 3, 3, 1, 4),
    UIBArgs(192, 3, 5, 2, 4),
    *[UIBArgs(192, 3, 3, 1, 4)] * 3,
    UIBArgs(192, 3, 5, 1, 4),
    *[UIBArgs(192, 5, 3, 1, 4)] * 5,
    UIBArgs(192, 3, 0, 1, 4),
    UIBArgs(512, 5, 5, 2, 4),
    *[UIBArgs(512, 5, 5, 1, 4)] * 3,
    UIBArgs(512, 5, 0, 1, 4),
    UIBArgs(512, 5, 3, 1, 4),
    *[UIBArgs(512, 5, 0, 1, 4)] * 2,
    UIBArgs(512, 5, 3, 1, 4),
    UIBArgs(512, 5, 5, 1, 4),
    *[UIBArgs(512, 5, 0, 1, 4)] * 3,
)

# paper does not give MQA heads, key/value dims or layer scale
HYBRID_MEDIUM = (
    ConvArgs(32, 3, 2),
    FusedIBArgs(48, 3, 2, 4),
    UIBArgs(80, 3, 5, 2, 4),
    UIBArgs(80, 3, 3, 1, 2),
    UIBArgs(160, 3, 5, 2, 6),
    # paper: ExtraDW, UIBArgs(160, 3, 3, 1, 4)
    UIBArgs(160, 0, 0, 1, 2),
    UIBArgs(160, 3, 3, 1, 4),
    UIBArgs(160, 3, 5, 1, 4),
    MQAArgs(4, 64, 64, 2),
    UIBArgs(160, 3, 3, 1, 4),
    MQAArgs(4, 64, 64, 2),
    UIBArgs(160, 3, 0, 1, 4),
    MQAArgs(4, 64, 64, 2),
    # paper: FFN, UIBArgs(160, 0, 0, 1, 4)
    UIBArgs(160, 3, 3, 1, 4),
    MQAArgs(4, 64, 64, 2),
    UIBArgs(160, 3, 0, 1, 4),
    UIBArgs(256, 5, 5, 2, 6),
    UIBArgs(256, 5, 5, 1, 4),
    *[UIBArgs(256, 3, 5, 1, 4)] * 2,
    # paper: FFN, UIBArgs(256, 0, 0, 1, 4)
    UIBArgs(256, 0, 0, 1, 2),
    # paper: ConvNext, UIBArgs(256, 3, 0, 1, 4)
    UIBArgs(256, 3, 5, 1, 2),
    # paper: ExtraDW, UIBArgs(256, 3, 5, 1, 2)
    UIBArgs(256, 0, 0, 1, 2),
    # paper: no such block
    UIBArgs(256, 0, 0, 1, 4),
    MQAArgs(4, 64, 64, 1),
    # paper: ExtraDW, UIBArgs(256, 5, 5, 1, 4)
    UIBArgs(256, 3, 0, 1, 4),
    MQAArgs(4, 64, 64, 1),
    # paper: FFN, UIBArgs(256, 0, 0, 1, 4)
    UIBArgs(256, 5, 5, 1, 4),
    MQAArgs(4, 64, 64, 1),
    # paper: FFN, UIBArgs(256, 0, 0, 1, 4)
    UIBArgs(256, 5, 0, 1, 4),
    MQAArgs(4, 64, 64, 1),
    UIBArgs(256, 5, 0, 1, 4),
)

HYBRID_LARGE = (
    ConvArgs(24, 3, 2),
    FusedIBArgs(48, 3, 2, 4),
    UIBArgs(96, 3, 5, 2, 4),
    UIBArgs(96, 3, 3, 1, 4),
    UIBArgs(192, 3, 5, 2, 4),
    *[UIBArgs(192, 3, 3, 1, 4)] * 3,
    UIBArgs(192, 3, 5, 1, 4),
    *[UIBArgs(192, 5, 3, 1, 4)] * 2,
    *[MQAArgs(8, 48, 48, 2), UIBArgs(192, 5, 3, 1, 4)] * 3,
    MQAArgs(8, 48, 48, 2),
    UIBArgs(192, 3, 0, 1, 4),
    UIBArgs(512, 5, 5, 2, 4),
    *[UIBArgs(512, 5, 5, 1, 4)] * 3,
    UIBArgs(512, 5, 0, 1, 4),
    UIBArgs(512, 5, 3, 1, 4),
    *[UIBArgs(512, 5, 0, 1, 4)] * 2,
    UIBArgs(512, 5, 3, 1, 4),
    UIBArgs(512, 5, 5, 1, 4),
    *[MQAArgs(8, 64, 64, 1), UIBArgs(512, 5, 0, 1, 4)] * 4,
)


class MobileNetV4(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: Literal[
            "conv_small",
            "conv_medium",
            "conv_large",
            "hybrid_medium",
            "hybrid_large",
        ] = "conv_medium",
        config: tuple[LAYER_ARGS] | None = None,
    ):
        """MobileNet version 4 [2404.10518]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            key (Literal["conv_small", "conv_medium", "conv_large", "hybrid_medium", "hybrid_large"], optional): model variant. Defaults to "conv_medium".
            config (tuple[LAYER_ARGS], optional): layer config, first layer is the stem and must be ConvArgs. Defaults to None.
        """
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return 1280

    @property
    def config(self) -> tuple[LAYER_ARGS]:
        return (
            self._config
            or {
                "conv_small": CONV_SMALL,
                "conv_medium": CONV_MEDIUM,
                "conv_large": CONV_LARGE,
                "hybrid_medium": HYBRID_MEDIUM,
                "hybrid_large": HYBRID_LARGE,
            }[self.key]
        )

    @property
    def activation(self) -> type[nn.Module]:
        return nn.GELU if self.key == "hybrid_large" else nn.ReLU

    def build_foot(self, input_channel: int, **kwargs):
        stem: ConvArgs = self.config[0]
        self.foot = Conv2dNormActivation(
            input_channel,
            stem.output_dim,
            stem.kernel,
            stem.stride,
            activation_layer=self.activation,
            inplace=self.activation is nn.ReLU or None,
        )

    def build_blocks(self, **kwargs):
        in_c = self.config[0].output_dim

        self.blocks = nn.Sequential()
        for layer_args in self.config[1:]:
            if isinstance(layer_args, ConvArgs):
                layer = Conv2dNormActivation(
                    in_c,
                    layer_args.output_dim,
                    layer_args.kernel,
                    layer_args.stride,
                    activation_layer=self.activation,
                    inplace=self.activation is nn.ReLU or None,
                )
            elif isinstance(layer_args, FusedIBArgs):
                layer = FusedInvertedBottleneck(
                    in_c,
                    layer_args.output_dim,
                    layer_args.kernel,
                    layer_args.stride,
                    layer_args.expansion_ratio,
                    activation=self.activation,
                    inplace=self.activation is nn.ReLU or None,
                )

                if in_c == layer_args.output_dim and layer_args.stride == 1:
                    layer = ResBlock(
                        in_c,
                        layer_args.output_dim,
                        block=layer,
                        stride=layer_args.stride,
                        activation=None,
                    )
            elif isinstance(layer_args, UIBArgs):
                layer = UniversalInvertedBottleneck(
                    in_c,
                    layer_args.output_dim,
                    (layer_args.start_dw_kernel, layer_args.middle_dw_kernel),
                    layer_args.stride,
                    layer_args.expansion_ratio,
                    activation=self.activation,
                    inplace=self.activation is nn.ReLU or None,
                    enable_layer_scale=self.key.startswith("hybrid"),
                )

                if in_c == layer_args.output_dim and layer_args.stride == 1:
                    layer = ResBlock(
                        in_c,
                        layer_args.output_dim,
                        block=layer,
                        stride=layer_args.stride,
                        activation=None,
                    )
            elif isinstance(layer_args, MQAArgs):
                layer = MobileMQA(
                    in_c,
                    layer_args.num_head,
                    layer_args.key_dim,
                    layer_args.value_dim,
                    layer_args.kv_stride,
                    enable_layer_scale=self.key.startswith("hybrid"),
                )
            else:
                raise TypeError(f"unknown layer config {layer_args}")

            self.blocks.append(layer)

            if not isinstance(layer_args, MQAArgs):
                in_c = layer_args.output_dim

    def build_neck(self, **kwargs):
        in_c = next(
            layer_args.output_dim
            for layer_args in reversed(self.config)
            if not isinstance(layer_args, MQAArgs)
        )
        self.neck = Conv2dNormActivation(
            in_c,
            960,
            1,
            activation_layer=self.activation,
            inplace=self.activation is nn.ReLU or None,
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dNormActivation(
                960,
                self.feature_dim,
                1,
                activation_layer=self.activation,
                inplace=self.activation is nn.ReLU or None,
            ),
            nn.Flatten(),
            nn.Linear(self.feature_dim, output_size),
        )
