import torch
from torch import nn
from torchlake.common.models import VggFeatureExtractor

from .network import UpSampling


class Fcn(nn.Module):
    def __init__(self, num_class: int):
        super(Fcn, self).__init__()
        self.backbone = VggFeatureExtractor("vgg16", "maxpool")
        self.up1 = UpSampling(512, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_class, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x, ["1_1", "2_1", "3_1", "4_1", "5_1"])

        score = self.up1(features[-1])
        score = score + features[-2]
        score = self.up2(score)
        score = score + features[-3]
        score = self.up3(score)
        score = score + features[-4]
        score = self.up4(score)
        score = score + features[-5]
        score = self.up5(score)
        return score
