from torch import Tensor, nn

from ..base.network import ConvBlock


class Extraction(nn.Module):
    def __init__(self):
        super(Extraction, self).__init__()
        self.conv_1 = nn.Sequential(
            ConvBlock(3, 64, 7, stride=2),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_2 = nn.Sequential(
            ConvBlock(64, 192, 3),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_3 = nn.Sequential(
            ConvBlock(192, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 256, 1),
            ConvBlock(256, 512, 3),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_4 = nn.Sequential(
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, 3),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.conv_5 = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),  # end of backbone
        )

        self.head = nn.Sequential(
            nn.Linear(1024, 1000),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.conv_4(y)
        y = self.conv_5(y)

        y = y.mean((2, 3))
        y = self.head(y)

        return y
