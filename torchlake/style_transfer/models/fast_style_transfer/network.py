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
        padding: int = 0,
        enable_in: bool = True,
        enable_relu: bool = True,
        enable_deconv: bool = False,
    ):
        super(ConvBlock, self).__init__()
        self.enable_in = enable_in
        self.enable_relu = enable_relu
        self.enable_deconv = enable_deconv

        self.reflection_pad = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel, stride, padding)

        if enable_in:
            self.instance_norm = nn.InstanceNorm2d(output_channel, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_deconv:
            x = F.interpolate(x, scale_factor=2)
        y = self.reflection_pad(x)
        y = self.conv(y)
        if self.enable_in:
            y = self.instance_norm(y)
        if self.enable_relu:
            y = F.relu(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, channel: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channel, channel, 3)
        self.conv2 = ConvBlock(channel, channel, 3, enable_relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y
