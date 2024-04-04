from torch import Tensor, nn

from ..base.network import ConvBlock


class BottleNeck(nn.Module):
    """K: 3 -> 1 -> 3
    C: c -> c/2 -> c
    """

    def __init__(self, channel: int, block_num: int):
        super(BottleNeck, self).__init__()
        self.conv = []
        for block_idx in range(block_num):
            channels = [channel // 2, channel]
            kernel = 3
            if block_idx % 2 != 0:
                channels.reverse()
                kernel = 1
            in_channels, out_channels = channels
            self.conv += [ConvBlock(in_channels, out_channels, kernel)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        return y


class ReorgLayer(nn.Module):
    def __init__(self, stride: int):
        """Stack patch

        Args:
            stride (int): stride
        """
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = x.reshape(
            b,
            c * self.stride * self.stride,
            h // self.stride,
            w // self.stride,
        )
        return x


class Darknet19(nn.Module):
    def __init__(self):
        """Darknet19"""
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
        self.head = nn.Sequential(nn.Linear(1024, 1000))

    def forward(self, x: Tensor) -> Tensor:
        # 112 x 112 x 32 # conv stride 2 will reduce hw 2 times too
        y = self.conv_1(x)
        y = self.conv_2(y)  # 64 x 56 x 56
        y = self.conv_3(y)  # 128 x 28 x 28
        y = self.conv_4(y)  # 256 x 14 x 14
        # 512 x 7 x 7 # conv stride 2 will reduce hw 2 times too
        y = self.conv_5(y)
        y = self.conv_6(y)  # 1024 x 7 x 7

        y = y.mean(dim=(2, 3))  # 1000
        y = self.head(y)

        return y
