import torch
import torch.nn as nn
from torch import Tensor

from ..base.detector import DetectorBase
from ..base.network import ConvBlock


class Yolov1(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_boxes: int,
        num_classes: int,
        finetune_weight: str = "",
    ):
        super(Yolov1, self).__init__()
        self.load_backbone(backbone, finetune_weight)

        self.output_size = (-1, num_classes + num_boxes * 5, 7, 7)

        self.conv_6 = nn.Sequential(
            ConvBlock(1024, 1024, 3),
            ConvBlock(1024, 1024, 3, stride=2),
            ConvBlock(1024, 1024, 3),
            ConvBlock(1024, 1024, 3),
        )

        self.head = nn.Sequential(
            ConvBlock(1024, 256, 3),
            nn.Dropout2d(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 7 * 7 * (num_classes + num_boxes * 5)),
        )

        for layer in self.conv_6.children():
            torch.nn.init.kaiming_normal_(layer.conv.conv.weight)

        for layer in self.head.children():
            torch.nn.init.kaiming_normal_(layer.conv.conv.weight)
            break

    def load_backbone(self, backbone: nn.Module, finetune_weight: str = ""):
        if finetune_weight:
            backbone.load_state_dict(torch.load(finetune_weight))
        backbone.head = nn.Identity()

        self.backbone = backbone

    def forward(self, x: Tensor) -> Tensor:
        for i in range(1, 6):
            x = getattr(self.backbone, f"conv_{i}")(x)

        y = self.conv_6(x)
        y: torch.Tensor = self.head(y)
        y = y.reshape(*self.output_size)

        return y


class Yolov1Resnet(DetectorBase):
    def __init__(
        self,
        num_layer: int,
        num_boxes: int,
        num_classes: int,
        finetune_weight: str = "",
    ):
        self.assert_num_layer(num_layer)
        super(Yolov1Resnet, self).__init__(
            finetune_weight,
            {"name": f"resnet{num_layer}"},
        )

        # like paper, add 4 convs after backbone
        self.conv = nn.Sequential(
            ConvBlock(512, 1024, 3, enable_relu=False),
            ConvBlock(1024, 1024, 3, stride=2, enable_relu=False),
            ConvBlock(1024, 1024, 3, enable_relu=False),
            ConvBlock(1024, 1024, 3, enable_relu=False),
        )

        self.head = nn.Sequential(nn.Conv2d(1024, 5 * num_boxes + num_classes, 1))

    def assert_num_layer(self, num_layer: int) -> nn.Module:
        if num_layer not in [18, 34, 50]:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone.conv1(x)
        y = self.backbone.bn1(y)
        y = self.backbone.relu(y)
        y = self.backbone.maxpool(y)
        y = self.backbone.layer1(y)
        y = self.backbone.layer2(y)
        y = self.backbone.layer3(y)
        y = self.backbone.layer4(y)

        y = self.conv(y)
        y = self.head(y)

        return y
