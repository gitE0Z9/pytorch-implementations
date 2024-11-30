import torch
from torch import nn
from torchlake.common.models import DepthwiseSeparableConv2d
from torchvision.ops import Conv2dNormActivation


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
        super().__init__()

        self.layers = nn.Sequential(
            # head layer
            Conv2dNormActivation(
                input_channel,
                int(32 * width_multiplier),
                stride=2,
            ),
            # middle layers
            *self.build_middle_layers(width_multiplier),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(int(1024 * width_multiplier), output_size)

    def build_middle_layers(
        self,
        width_multiplier: float = 1,
    ) -> list[nn.Module]:
        """Build middle layers.

        Args:
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
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

        return middle_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        return self.fc(y)
