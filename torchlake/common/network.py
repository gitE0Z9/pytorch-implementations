import torch
from torch import nn
from .constants import IMAGENET_MEAN, IMAGENET_STD


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        group: int = 1,
        enable_bn: bool = True,
        enable_relu: bool = True,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            stride,
            padding,
            dilation,
            group,
            bias=not enable_bn,
        )
        self.bn = nn.BatchNorm2d(output_channel) if enable_bn else enable_bn
        self.relu = nn.ReLU(True) if enable_relu else enable_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        if self.relu:
            y = self.relu(y)

        return y


class ImageNormalization(nn.Module):
    def __init__(
        self,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
    ):
        super(ImageNormalization, self).__init__()
        ## C,1,1 shape for broadcasting
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, img: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        original_shape = img.size()

        if img.dim() == 3:
            img = img.unsqueeze(0)

        if not reverse:
            normalized = (img - self.mean.to(img.device)) / self.std.to(img.device)
        else:
            normalized = img * self.std.to(img.device) + self.mean.to(img.device)

        return normalized.reshape(*original_shape)
