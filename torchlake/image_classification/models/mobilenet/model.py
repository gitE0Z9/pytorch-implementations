import torch
from torch import nn
from torchlake.common.network import ConvBnRelu, DepthwiseSeparableConv2d


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
            DepthwiseSeparableConv2d(
                int(32 * width_multiplier),
                int(64 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(64 * width_multiplier),
                int(128 * width_multiplier),
                3,
                stride=2,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(128 * width_multiplier),
                int(128 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(128 * width_multiplier),
                int(256 * width_multiplier),
                3,
                stride=2,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(256 * width_multiplier),
                int(256 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(256 * width_multiplier),
                int(512 * width_multiplier),
                3,
                stride=2,
                padding=1,
            ),
            # repeating 5 times
            DepthwiseSeparableConv2d(
                int(512 * width_multiplier),
                int(512 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(512 * width_multiplier),
                int(512 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(512 * width_multiplier),
                int(512 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(512 * width_multiplier),
                int(512 * width_multiplier),
                3,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(512 * width_multiplier),
                int(512 * width_multiplier),
                3,
                padding=1,
            ),
            # final layers
            DepthwiseSeparableConv2d(
                int(512 * width_multiplier),
                int(1024 * width_multiplier),
                3,
                stride=2,
                padding=1,
            ),
            DepthwiseSeparableConv2d(
                int(1024 * width_multiplier),
                int(1024 * width_multiplier),
                3,
                stride=2,
                padding=1,
            ),
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
