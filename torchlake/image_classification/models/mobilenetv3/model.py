from typing import Literal

from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.model_base import ModelBase

from .network import InvertedResidualBlockV3


class MobileNetV3(ModelBase):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        key: Literal["small", "large"] = "large",
        config: tuple[tuple[int]] | None = None,
    ):
        """MobileNet version 3 [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            key (Literal["small", "large"], optional): small or large. Defaults to "large".
            config (tuple[tuple[int]], optional): layer conifg, format is (input_channel, output_channel, kernel, stride, expansion_size, hard_swish, enable_se). Defaults to None.
        """
        self.key = key
        self._config = config
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self):
        return 1280 if self.key == "large" else 1024

    @property
    def config(self):
        # input_channel, output_channel, kernel, stride, expansion_size, hard_swish, enable_se
        return (
            self._config
            or {
                "small": (
                    (16, 16, 3, 2, 16, False, True),
                    (16, 24, 3, 2, 72, False, False),
                    (24, 24, 3, 1, 88, False, False),
                    (24, 40, 5, 2, 96, True, True),
                    (40, 40, 5, 1, 240, True, True),
                    (40, 40, 5, 1, 240, True, True),
                    (40, 48, 5, 1, 120, True, True),
                    (48, 48, 5, 1, 144, True, True),
                    (48, 96, 5, 2, 288, True, True),
                    (96, 96, 5, 1, 576, True, True),
                    (96, 96, 5, 1, 576, True, True),
                ),
                "large": (
                    (16, 16, 3, 1, 16, False, False),
                    (16, 24, 3, 2, 64, False, False),
                    (24, 24, 3, 1, 72, False, False),
                    (24, 40, 5, 2, 72, False, True),
                    (40, 40, 5, 1, 120, False, True),
                    (40, 40, 5, 1, 120, False, True),
                    (40, 80, 3, 2, 240, True, False),
                    (80, 80, 3, 1, 200, True, False),
                    (80, 80, 3, 1, 184, True, False),
                    (80, 80, 3, 1, 184, True, False),
                    (80, 112, 3, 1, 480, True, True),
                    (112, 112, 3, 1, 672, True, True),
                    (112, 160, 5, 2, 672, True, True),
                    (160, 160, 5, 1, 960, True, True),
                    (160, 160, 5, 1, 960, True, True),
                ),
            }[self.key]
        )

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                16,
                stride=2,
                activation_layer=nn.Hardswish,
                inplace=False,
            ),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()
        for (
            in_c,
            out_c,
            kernel,
            stride,
            expansion_size,
            hard_swish,
            enable_se,
        ) in self.config:
            self.blocks.append(
                InvertedResidualBlockV3(
                    in_c,
                    out_c,
                    kernel,
                    stride=stride,
                    expansion_size=expansion_size,
                    enable_relu=not hard_swish,
                    enable_se=enable_se,
                )
            )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            Conv2dNormActivation(
                self.config[-1][0],
                self.config[-1][4],
                1,
                activation_layer=nn.Hardswish,
                inplace=False,
            ),
            FlattenFeature(),
            nn.Linear(self.config[-1][4], self.feature_dim),
            nn.Hardswish(),
            nn.Linear(self.feature_dim, output_size),
        )
