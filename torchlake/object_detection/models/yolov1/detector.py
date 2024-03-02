import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from .network import Conv3x3, Extraction


class Yolov1(nn.Module):
    def __init__(self):
        super(Yolov1, self).__init__()
        self.backbone = Extraction(pretrained=True)
        self.backbone.head = nn.Identity()

        self.conv_6 = nn.Sequential(
            Conv3x3(1024, 1024),
            Conv3x3(1024, 1024, stride=2),
            Conv3x3(1024, 1024),
            Conv3x3(1024, 1024),
        )

        self.conv_out = nn.Sequential(nn.Conv2d(1024, 30, (1, 1), stride=1))

        # too much parameters, over one fourth billion

    #         self.classifier = nn.Sequential(
    #             nn.Linear(7*7*512,1024),
    #             nn.Dropout(p=.5),
    #             nn.LeakyReLU(0.1),
    #             nn.Linear(1024,7*7*30),
    #         )

    def forward(self, x: Tensor) -> Tensor:
        for i in range(1, 7):
            x = getattr(self.backbone, f"conv_{i}")(x)

        x = self.conv_6(x)
        x = self.conv_out(x)

        # tmp = torch.flatten(tmp,start_dim=1)
        # tmp = self.classifier(tmp)
        # tmp = tmp.reshape(-1,30,7,7)#.contiguous()

        return x


class Yolov1Resnet(nn.Module):
    def __init__(
        self,
        layer_number: int,
        num_boxes: int,
        num_classes: int,
        finetune_weight: str = "",
    ):
        super(Yolov1Resnet, self).__init__()

        self.load_backbone(layer_number, finetune_weight)

        # as paper suggested, deeper conv layer
        # convhead-512 no first layer and all 512
        self.conv = nn.Sequential(
            Conv3x3(512, 1024, enable_bn=True, enable_relu=True),
            Conv3x3(1024, 1024, stride=2, enable_bn=True, enable_relu=True),
            Conv3x3(1024, 1024, enable_bn=True, enable_relu=True),
            Conv3x3(1024, 1024, enable_bn=True, enable_relu=True),
        )

        # self.conv = self.backbone._make_layer(BasicBlock, 1024, 2, stride=2)

        # too much parameters, over one fourth billion
        #         self.classifier = nn.Sequential(
        #             nn.Linear(7*7*512,1024),
        #             nn.Dropout(p=.5),
        #             nn.LeakyReLU(0.1),
        #             nn.Linear(1024,7*7*30),
        #         )

        self.conv_out = nn.Sequential(nn.Conv2d(1024, 5 * num_boxes + num_classes, 1))

    def build_backbone(self, num_layer: int) -> nn.Module:
        if num_layer not in [18, 34, 50]:
            raise NotImplementedError

        return torchvision.models.get_model(f"resnet{num_layer}", weights="DEFAULT")

    def load_backbone(self, num_layer: int, finetune_weight: str = ""):
        backbone = self.build_backbone(num_layer)

        if finetune_weight:
            backbone.load_state_dict(torch.load(finetune_weight))
        backbone.fc = nn.Identity()

        self.backbone = backbone

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
        y = self.conv_out(y)

        # tmp = torch.flatten(tmp,start_dim=1)
        # tmp = self.classifier(tmp)
        # tmp = tmp.reshape(-1,30,7,7)#.contiguous()

        return y
