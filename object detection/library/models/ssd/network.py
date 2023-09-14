import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.vgg import VGG16_Weights


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
        output = self.weight * x
        return output


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
    def __init__(self, input_channel: int, num_classes: int, num_priors: int):
        super(RegHead, self).__init__()
        self.loc = nn.Conv2d(
            input_channel,
            num_priors * 4,
            kernel_size=3,
            padding=1,
        )
        self.conf = nn.Conv2d(
            input_channel,
            num_priors * num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loc(x), self.conf(x)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.feature_extractor = torchvision.models.vgg16(
            weights=VGG16_Weights.DEFAULT
        ).features  # [:30]
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature_list = []

        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i == 22:
                feature_list.append(self.norm(x))
            if i == 29:
                break

        x = self.pool5(x)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        feature_list.append(x)

        x = self.conv8(x)
        feature_list.append(x)

        x = self.conv9(x)
        feature_list.append(x)

        x = self.conv10(x)
        feature_list.append(x)

        x = self.conv11(x)
        feature_list.append(x)

        return feature_list
