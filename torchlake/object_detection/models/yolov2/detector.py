import torch
import torch.nn as nn
import torchvision

from .network import ConvBlock, ReorgLayer


class Yolov2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_anchors: int,
        num_classes: int,
        finetune_weight: str = "",
    ):
        super(Yolov2, self).__init__()
        self.backbone = backbone
        if finetune_weight:
            self.backbone.load_state_dict(torch.load(finetune_weight))
        self.backbone.head = nn.Identity()  # save disk space

        self.conv1 = nn.Sequential(
            ConvBlock(1024, 1024, 3),
            ConvBlock(1024, 1024, 3),
        )

        self.passthrough = nn.Sequential(
            ConvBlock(512, 512 // 8, 1),
            ReorgLayer(2),
        )

        self.head = nn.Sequential(
            ConvBlock(1280, 1024, 3),
            nn.Conv2d(1024, num_anchors * (num_classes + 5), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i in range(6):
            conv_layer = getattr(self.backbone, f"conv_{i+1}")
            x = conv_layer(x)
            if i + 1 in [5, 6]:
                features.append(x)

        skip, x = features
        skip = self.passthrough(skip)  # 13 x 13 x 64*4

        y = self.conv1(x)  # 13 x 13 x 1024
        y = torch.cat([y, skip], dim=1)  # 13 x 13 x 1280
        y = self.head(y)  # 13 x 13 x C+5

        return y


class Yolov2Resnet(nn.Module):
    def __init__(
        self,
        layer_number: int,
        num_anchors: int,
        num_classes: int,
        finetune_weight: str = "",
    ):
        super(Yolov2Resnet, self).__init__()
        self.load_backbone(layer_number, finetune_weight)

        in_ch = 512 if layer_number != 50 else 2048
        self.conv1 = nn.Sequential(
            ConvBlock(in_ch, 1024, 3),
            ConvBlock(1024, 1024, 3),
        )

        pt_ch = 256 if layer_number != 50 else 1024
        self.passthrough = nn.Sequential(
            ConvBlock(pt_ch, pt_ch // 8, 1),
            ReorgLayer(2),
        )

        concat_ch = 1024 + pt_ch // 8 * 4
        self.head = nn.Sequential(
            ConvBlock(concat_ch, 1024, 3),
            nn.Conv2d(1024, num_anchors * (num_classes + 5), 1),
        )

    def build_backbone(self, layer_number: int) -> nn.Module:
        if layer_number not in [18, 34, 50]:
            raise NotImplementedError

        backbone_class = getattr(torchvision.models, f"resnet{layer_number}")
        backbone = backbone_class(weights="DEFAULT")

        return backbone

    def load_backbone(self, layer_number: int, finetune_weight: str = ""):
        backbone = self.build_backbone(layer_number)

        if finetune_weight:
            backbone.load_state_dict(torch.load(finetune_weight))
        backbone.fc = nn.Identity()

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone.conv1(x)
        y = self.backbone.bn1(y)
        y = self.backbone.relu(y)
        y = self.backbone.maxpool(y)
        y = self.backbone.layer1(y)
        y = self.backbone.layer2(y)
        y = self.backbone.layer3(y)  # 26 x 26 x 256

        skip = self.passthrough(y)  # 13 x 13 x 32*4

        y = self.backbone.layer4(y)  # 13 x 13 x 512
        y = self.conv1(y)  # 13 x 13 x 1024
        y = torch.cat([y, skip], dim=1)  # 13 x 13 x 1152
        y = self.head(y)

        return y
