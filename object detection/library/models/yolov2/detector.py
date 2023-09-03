import torch
import torch.nn as nn
import torchvision
from models.yolov2.network import ConvBlock, ReorgLayer


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
        f_list = []
        for i in range(6):
            c = getattr(self.backbone, f"conv_{i+1}")
            x = c(x)
            if i + 1 in [5, 6]:
                f_list.append(x)

        skip, x = f_list

        x = self.conv1(x)  # 13 x 13 x 1024

        skip = self.passthrough(skip)  # 13 x 13 x 64*4

        x = torch.cat([x, skip], dim=1)  # 13 x 13 x 1280

        x = self.head(x)

        return x


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

        backbone = getattr(torchvision.models, f"resnet{layer_number}")(weights=True)

        return backbone

    def load_backbone(self, layer_number: int, finetune_weight: str = ""):
        backbone = self.build_backbone(layer_number)

        if finetune_weight:
            backbone.load_state_dict(torch.load(finetune_weight))
        backbone.fc = nn.Identity()

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = self.backbone.conv1(x)
        tmp = self.backbone.bn1(tmp)
        tmp = self.backbone.relu(tmp)
        tmp = self.backbone.maxpool(tmp)
        tmp = self.backbone.layer1(tmp)
        tmp = self.backbone.layer2(tmp)
        tmp = self.backbone.layer3(tmp)  # 26 x 26 x 256

        skip = self.passthrough(tmp)  # 13 x 13 x 32*4

        tmp = self.backbone.layer4(tmp)  # 13 x 13 x 512
        tmp = self.conv1(tmp)  # 13 x 13 x 1024
        tmp = torch.cat([tmp, skip], dim=1)  # 13 x 13 x 1152
        tmp = self.head(tmp)

        return tmp
