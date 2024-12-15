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

EFFICIENTNET_V1_NAMES = Literal[
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "b7",
]


EFFICIENTNET_V2_NAMES = Literal[
    "s",
    "m",
    "l",
]

VIT_NAMES = Literal[
    "b16",
    "b32",
    "l16",
    "l32",
]
