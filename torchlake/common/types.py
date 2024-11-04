from typing import Literal
from torch import nn

CONV_TYPE = nn.Conv1d | nn.Conv2d | nn.Conv3d
BN_TYPE = nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d

RESNET_NAMES = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]
