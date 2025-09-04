import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        enable_in: bool = True,
        enable_relu: bool = True,
    ):
        super().__init__()
        self.enable_in = enable_in
        self.enable_relu = enable_relu

        self.reflection_pad = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel, stride)

        if enable_in:
            self.instance_norm = nn.InstanceNorm2d(output_channel, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reflection_pad(x)
        y = self.conv(y)
        if self.enable_in:
            y = self.instance_norm(y)
        if self.enable_relu:
            y = F.relu(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(channel, channel, 3),
            ConvBlock(channel, channel, 3, enable_relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
