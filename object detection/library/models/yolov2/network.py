from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """ConvBNReLU block"""

    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, (kernel, kernel), padding=kernel // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.bn(tmp)
        tmp = F.leaky_relu(tmp, 0.1, inplace=True)
        return tmp


class BottleNeck(nn.Module):
    """K: 3 -> 1 -> 3
    C: c -> c/2 -> c
    """

    def __init__(self, channel: int, block_num: int):
        super(BottleNeck, self).__init__()
        self.conv = []
        for b in range(block_num):
            tmp = [channel // 2, channel]
            k = 3
            if b % 2 != 0:
                tmp.reverse()
                k = 1
            in_channels, out_channels = tmp
            self.conv += [ConvBlock(in_channels, out_channels, k)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        tmp = self.conv(x)
        return tmp


class Darknet19(nn.Module):
    """darknet19"""

    def __init__(self):
        super(Darknet19, self).__init__()
        self.conv_1 = nn.Sequential(
            ConvBlock(3, 32, 3),
        )
        self.conv_2 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            ConvBlock(32, 64, 3),
        )
        self.conv_3 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            BottleNeck(128, 3),
        )
        self.conv_4 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            BottleNeck(256, 3),
        )
        self.conv_5 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            BottleNeck(512, 5),
        )
        self.conv_6 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            BottleNeck(1024, 5),
        )
        self.head = nn.Sequential(nn.Conv2d(1024, 1000, (1, 1)))

    def forward(self, x):
        # 112 x 112 x 32 # conv stride 2 will reduce hw 2 times too
        tmp = self.conv_1(x)
        tmp = self.conv_2(tmp)  # 56 X 56 X 64
        tmp = self.conv_3(tmp)  # 28 x 28 x 128
        tmp = self.conv_4(tmp)  # 14 x 14 x 256
        # 7 x 7 x 512 # conv stride 2 will reduce hw 2 times too
        tmp = self.conv_5(tmp)
        tmp = self.conv_6(tmp)  # 7 x 7 x 1024

        tmp = self.head(tmp)
        tmp = tmp.mean(dim=(2, 3))

        return tmp


class ReorgLayer(nn.Module):
    def __init__(self, stride: int):
        """Stack patch

        Args:
            stride (int): stride
        """
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(
            b,
            c * self.stride * self.stride,
            h // self.stride,
            w // self.stride,
        )
        return x
