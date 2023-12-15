import torch
from torch import nn
from .network import DownSampling, UpSampling, ConvInRelu


class Pix2PixDiscriminator(nn.Module):
    def __init__(self):
        super(Pix2PixDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ConvInRelu(6, 32),
            ConvInRelu(32, 64),
            ConvInRelu(64, 128),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, y], dim=1)
        z = self.discriminator(z)
        return z


class Pix2PixGenerator(nn.Module):
    def __init__(self, output_size: int):
        """Pix2pix generator, a UNet

        Args:
            output_size (int): the channel size of ouput
        """
        super(Pix2PixGenerator, self).__init__()
        self.down1 = DownSampling(3, 64)  # 64,112
        self.down2 = DownSampling(64, 128)  # 128,56
        self.down3 = DownSampling(128, 256)  # 256,28
        self.down4 = DownSampling(256, 512)  # 512,14
        self.expand = DownSampling(512, 1024)  # 1024,7
        self.up1 = UpSampling(1024, 1024)  # 512+512,14
        self.up2 = UpSampling(1024, 512)  # 256+256,28
        self.up3 = UpSampling(512, 256)  # 128+128,56
        self.up4 = UpSampling(256, 128)  # 64+64,112
        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 64, 224
        self.conv = nn.Conv2d(64, output_size, 1)  # 20, 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        intermediate = self.expand(d4)
        u1 = self.up1(intermediate, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4)
        y = self.conv(u5)
        return y
