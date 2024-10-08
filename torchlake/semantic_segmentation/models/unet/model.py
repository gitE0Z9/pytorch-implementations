import torch
from torch import nn
from .network import DoubleConv, DownSampling, UpSampling


class UNet(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self.convhead = DoubleConv(3, 64)  # 64, x
        self.down1 = DownSampling(64, 128)  # 128, x/2
        self.down2 = DownSampling(128, 256)  # 256, x/4
        self.down3 = DownSampling(256, 512)  # 512, x/8
        self.down4 = DownSampling(512, 1024)  # 1024, x/16
        self.up1 = UpSampling(1024, 512)  # 512+512,x/8
        self.up2 = UpSampling(512, 256)  # 256+256,x/4
        self.up3 = UpSampling(256, 128)  # 128+128,x/2
        self.up4 = UpSampling(128, 64)  # 64+64,x
        self.convend = nn.Conv2d(64, num_class, 1)  # 20, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convhead(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x)
        out = self.convend(u4)
        return out
