from itertools import product
from math import sqrt

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class PriorBox:
    def __init__(self):
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.image_size = IMAGE_SIZE
        self.min_scale = 0.2
        self.max_scale = 0.9
        self.anchor_num = [4, 6, 6, 6, 4, 4]
        self.aspect_ratios = [1, 2, 3, 1 / 2, 1 / 3]

    def build_anchors(self) -> torch.Tensor:
        placeholder = []
        for k, size in enumerate(self.feature_maps):
            sk = self.min_scale + (self.max_scale - self.min_scale) * k / (
                len(self.feature_maps) - 1
            )
            sk_1 = self.min_scale + (self.max_scale - self.min_scale) * (k + 1) / (
                len(self.feature_maps) - 1
            )
            sk_prime = sqrt(sk * sk_1)
            for i, j in product(range(size), repeat=2):  # mesh grid y,x index
                cx = (j + 0.5) / size
                cy = (i + 0.5) / size

                for ar in self.aspect_ratios:
                    if (
                        self.anchor_num[k] == 4 and ar in [1, 2, 1 / 2]
                    ) or self.anchor_num[k] == 6:
                        placeholder.append([cx, cy, sk * sqrt(ar), sk / sqrt(ar)])
                        if ar == 1:
                            # ar = 1, two anchors
                            placeholder.append([cx, cy, sk_prime, sk_prime])

        output = torch.Tensor(placeholder)
        output.clamp_(0, 1)
        return output


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.feature_extractor = torchvision.models.vgg16(weights=True).features[:30]
        self.feature_extractor[16] = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True
        )
        self.norm = L2Norm(512, 20)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(
            512, 1024, kernel_size=3, padding=6, dilation=6
        )  # dilation // kernel_size
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv8 = ExtraConv(1024, 256, 512, stride=2, padding=1)
        self.conv9 = ExtraConv(512, 128, 256, stride=2, padding=1)
        self.conv10 = ExtraConv(256, 128, 256, stride=1, padding=0)
        self.conv11 = ExtraConv(256, 128, 256, stride=1, padding=0)

    def forward(self, x) -> list[torch.Tensor]:
        f_list = []

        for i, l in enumerate(self.feature_extractor):
            x = l(x)
            if i == 22:
                f_list.append(self.norm(x))
            if i == 29:
                break

        x = self.pool5(x)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        f_list.append(x)

        x = self.conv8(x)
        f_list.append(x)

        x = self.conv9(x)
        f_list.append(x)

        x = self.conv10(x)
        f_list.append(x)

        x = self.conv11(x)
        f_list.append(x)

        return f_list


class L2Norm(nn.Module):
    def __init__(self, channels: int, scale: int):
        super(L2Norm, self).__init__()
        self.n_channels = channels
        self.gamma = scale * 1.0
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.full((1, self.n_channels, 1, 1), self.gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=1, p=2, keepdim=True) + self.eps
        x = x / norm
        out = self.weight * x
        return out


class ExtraConv(nn.Module):
    def __init__(
        self,
        input_channel: int,
        intermediate_channel: int,
        output_channel: int,
        stride: int,
        padding: int,
    ):
        super(ExtraConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channel,
                intermediate_channel,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                intermediate_channel,
                output_channel,
                kernel_size=3,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out


class RegHead(nn.Module):
    def __init__(self, input_channel: int, num_classes: int, multiplier: int):
        super(RegHead, self).__init__()
        self.loc = nn.Conv2d(
            input_channel,
            multiplier * 4,
            kernel_size=3,
            padding=1,
        )
        self.conf = nn.Conv2d(
            input_channel,
            multiplier * num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loc(x), self.conf(x)
