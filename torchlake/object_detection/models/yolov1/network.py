import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchlake.common.network import ConvBnRelu


class Conv3x3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        enable_bn: bool = False,
        enable_relu: bool = False,
    ):
        super(Conv3x3, self).__init__()
        self.conv = ConvBnRelu(
            in_channels,
            out_channels,
            kernel=3,
            stride=stride,
            padding=1,
            enable_bn=enable_bn,
            enable_relu=enable_relu,
        )

        if not enable_relu:
            self.activation = nn.LeakyReLU(0.1, True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if getattr(self, "activation", None):
            y = F.leaky_relu(y, 0.1, inplace=True)

        return y


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1, self).__init__()
        self.conv = ConvBnRelu(
            in_channels,
            out_channels,
            kernel=1,
            stride=stride,
            enable_bn=False,
            enable_relu=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        y = F.leaky_relu(y, 0.1, inplace=True)
        return y


class Extraction(nn.Module):
    def __init__(self):
        super(Extraction, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), stride=2, padding=3),
            #             nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_2 = nn.Sequential(
            Conv3x3(64, 192),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_3 = nn.Sequential(
            Conv1x1(192, 128),
            Conv3x3(128, 256),
            Conv1x1(256, 256),
            Conv3x3(256, 512),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_4 = nn.Sequential(
            Conv1x1(512, 256),
            Conv3x3(256, 512),
            Conv1x1(512, 256),
            Conv3x3(256, 512),
            Conv1x1(512, 256),
            Conv3x3(256, 512),
            Conv1x1(512, 256),
            Conv3x3(256, 512),
            Conv1x1(512, 512),
            Conv3x3(512, 1024),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_5 = nn.Sequential(
            Conv1x1(1024, 512),
            Conv3x3(512, 1024),
            Conv1x1(1024, 512),
            Conv3x3(512, 1024),  # end of backbone
        )

        self.head = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 1000),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.conv_4(y)
        y = self.conv_5(y)

        y = torch.mean(y, (2, 3))
        y = self.head(y)

        return y
