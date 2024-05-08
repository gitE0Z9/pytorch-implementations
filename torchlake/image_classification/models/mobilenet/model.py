import torch
from torch import nn
from torchlake.common.models import DepthwiseSeparableConv2d
from torchlake.common.network import ConvBnRelu

from .network import InvertedResidualBlock


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
            ConvBnRelu(
                input_channel,
                int(32 * width_multiplier),
                3,
                stride=2,
                padding=1,
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
            ConvBnRelu(
                input_channel,
                int(32 * width_multiplier),
                3,
                stride=2,
                padding=1,
            ),
            # middle layers
            *middle_layers,
            # final layers
            ConvBnRelu(
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
