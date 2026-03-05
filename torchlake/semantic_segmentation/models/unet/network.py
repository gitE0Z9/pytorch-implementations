import math

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop

from torchlake.semantic_segmentation.models.fcn.network import (
    init_deconv_with_bilinear_kernel,
)


class ConvBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2dNormActivation(input_channel, output_channel, 3),
            Conv2dNormActivation(output_channel, output_channel, 3),
        )

        for layer in self.layers:
            nn.init.kaiming_normal_(layer[0].weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownSampling(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(input_channel, output_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpSampling(nn.Module):
    def __init__(
        self,
        deep_input_channel: int,
        shallow_input_channel: int,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            deep_input_channel,
            shallow_input_channel,
            4,
            stride=2,
            padding=1,
        )
        init_deconv_with_bilinear_kernel(self.upsample)
        self.block = ConvBlock(2 * shallow_input_channel, shallow_input_channel)

    def forward(self, deep_x: torch.Tensor, shallow_x: torch.Tensor) -> torch.Tensor:
        y = self.upsample(deep_x)
        cropper = CenterCrop(shallow_x.shape[-2:])
        y = cropper(y)
        # y_pad, x_pad = shallow_x.size(3) - y.size(3), shallow_x.size(2) - y.size(2)
        # y = nn.ReflectionPad2d((0, 0, y_pad // 2, x_pad // 2))(y)
        y = torch.cat([shallow_x, y], dim=1)
        y = self.block(y)
        return y
