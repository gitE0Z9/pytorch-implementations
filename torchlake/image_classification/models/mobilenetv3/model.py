from typing import Literal

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import InvertedResidualBlockV3


class MobileNetV3(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        size: Literal["small", "large"] = "large",
    ):
        """MobileNet version 3 [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            size ("small|large", optional): small or large. Defaults to "large".
        """
        super().__init__()

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                16,
                stride=2,
                activation_layer=nn.Hardswish,
                inplace=False,
            ),
            # middle layers
            *self.build_middle_layers(size),
            # final layers
            Conv2dNormActivation(
                160 if size == "large" else 96,
                960 if size == "large" else 576,
                1,
                activation_layer=nn.Hardswish,
                inplace=False,
            ),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            Conv2dNormActivation(
                960 if size == "large" else 576,
                1280 if size == "large" else 1024,
                1,
                norm_layer=None,
                activation_layer=nn.Hardswish,
                inplace=False,
            ),
            Conv2dNormActivation(
                1280 if size == "large" else 1024,
                output_size,
                1,
                norm_layer=None,
                activation_layer=None,
            ),
            nn.Flatten(),
        )

    def build_middle_layers(
        self,
        size: Literal["small"] | Literal["large"] = "large",
    ) -> list[nn.Module]:
        """Build middle layers.

        Args:
            size ("small|large", optional): small or large. Defaults to "large".
        """
        configs = {
            # input_channel, output_channel, kernel, stride, expansion_size, hard_swish, enable_se
            "small": [
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
            ],
            "large": [
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
            ],
        }

        middle_layers = []
        for (
            in_c,
            out_c,
            kernel,
            stride,
            expansion_size,
            hard_swish,
            enable_se,
        ) in configs[size]:
            middle_layers.append(
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

        return middle_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)
