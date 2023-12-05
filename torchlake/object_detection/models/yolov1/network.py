import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (3, 3),
            stride=stride,
            padding=1,
        )

    #         self.bn = nn.BatchNorm2d(out_channels)
    #         nn.init.trunc_normal_(self.conv.weight,mean=0.0,std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        tmp = self.conv(x)
        #         tmp = self.bn(tmp)
        tmp = F.leaky_relu(tmp, 0.1, inplace=True)
        return tmp


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride)
        #         self.bn = nn.BatchNorm2d(out_channels)
        nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        tmp = self.conv(x)
        #         tmp = self.bn(tmp)
        tmp = F.leaky_relu(tmp, 0.1, inplace=True)
        return tmp


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

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 1000),
        )

    def forward(self, x: Tensor) -> Tensor:
        tmp = self.conv_1(x)
        tmp = self.conv_2(tmp)
        tmp = self.conv_3(tmp)
        tmp = self.conv_4(tmp)
        tmp = self.conv_5(tmp)

        tmp = torch.mean(tmp, (2, 3))
        tmp = self.classifier(tmp)

        return tmp
