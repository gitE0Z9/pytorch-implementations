from typing import Literal

import torch
from torch import nn
from torchlake.common.models import DepthwiseSeparableConv2d
from torchlake.common.models.se import SqueezeExcitation2d
from torchlake.common.network import ConvBnRelu
from torchvision.ops import Conv2dNormActivation

from .network import InvertedResidualBlock, InvertedResidualBlockV3


class MobileNetV1(nn.Module):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """MobileNet version 1 [1704.04861v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        super(MobileNetV1, self).__init__()
        # input_channel, output_channel, stride, num_layer
        configs = [
            (32, 64, 1, 1),
            (64, 128, 2, 1),
            (128, 128, 1, 1),
            (128, 256, 2, 1),
            (256, 256, 1, 1),
            (256, 512, 2, 1),
            (512, 512, 1, 5),
            (512, 1024, 2, 1),
            (1024, 1024, 2, 1),
        ]
        middle_layers = []
        for in_c, out_c, stride, num_layer in configs:
            middle_layers.extend(
                [
                    DepthwiseSeparableConv2d(
                        int(in_c * width_multiplier),
                        int(out_c * width_multiplier),
                        stride=stride,
                    )
                    for _ in range(num_layer)
                ]
            )

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                int(32 * width_multiplier),
                stride=2,
            ),
            # middle layers
            *middle_layers,
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(int(1024 * width_multiplier), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)


class MobileNetV2(nn.Module):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """MobileNet version 2 [1801.04381v4]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        super(MobileNetV2, self).__init__()

        # input_channel, output_channel, stride, expansion_factor, num_layer
        configs = [
            (32, 16, 1, 1, 1),
            (16, 24, 2, 6, 2),
            (24, 32, 2, 6, 3),
            (32, 64, 2, 6, 4),
            (64, 96, 1, 6, 3),
            (96, 160, 2, 6, 3),
            (160, 320, 1, 6, 1),
        ]
        middle_layers = []
        for (
            in_c,
            out_c,
            stride,
            expansion_ratio,
            num_layer,
        ) in configs:
            middle_layers.extend(
                [
                    InvertedResidualBlock(
                        int((in_c if l == 0 else out_c) * width_multiplier),
                        int(out_c * width_multiplier),
                        stride=stride if l == 0 else 1,
                        expansion_ratio=expansion_ratio,
                    )
                    for l in range(num_layer)
                ]
            )

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                int(32 * width_multiplier),
                stride=2,
            ),
            # middle layers
            *middle_layers,
            # final layers
            Conv2dNormActivation(
                int(320 * width_multiplier),
                int(1280 * width_multiplier) if width_multiplier > 1 else 1280,
                1,
            ),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                int(1280 * width_multiplier) if width_multiplier > 1 else 1280,
                output_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)


class MobileNetV3(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        size: Literal["small"] | Literal["large"] = "large",
    ):
        """MobileNet version 3 [1905.02244]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            size ("small|large", optional): small or large. Defaults to "large".
        """
        super(MobileNetV3, self).__init__()
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
            *middle_layers,
            # final layers
            ConvBnRelu(
                160 if size == "large" else 96,
                960 if size == "large" else 576,
                1,
                activation=nn.Hardswish(),
            ),
            SqueezeExcitation2d(
                960 if size == "large" else 576,
                reduction_ratio=4,
                activations=(nn.ReLU(True), nn.Hardsigmoid()),
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
            nn.Conv2d(1280 if size == "large" else 1024, output_size, 1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)
