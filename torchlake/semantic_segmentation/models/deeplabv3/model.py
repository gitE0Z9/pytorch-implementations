from copy import deepcopy
from functools import partial
from typing import Literal

import torch
from torch import nn
from torchlake.common.models import ResNetFeatureExtractor
from torchvision.transforms import CenterCrop
from torchvision.ops import Conv2dNormActivation

from .network import ASPP, CascadeASPP


class DeepLabV3(nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        dilations: list[int] = [6, 12, 18],
        backbone_name: Literal["resnet50", "resnet101", "resnet152"] = "resnet50",
        neck_type: Literal["parallel", "cascade"] = "parallel",
        frozen_backbone: bool = False,
    ):
        """DeepLab v3 in paper [1706.05587v3]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            dilations (list[int], optional): dilation size of ASPP, for 16x [6, 12, 18], for 8x [12, 24, 36]. Defaults to [6, 12, 18].
            backbone_name (Literal["resnet50", "resnet101", "resnet152"], optional): resnet network name. Defaults to "resnet50".
            neck_type: (Literal["parallel", "cascade"], optional): neck type is multi-grid cascade module or parallel ASPP. Defaults to "parallel",
            fronzen_backbone (bool, optional): froze the resnet backbone or not. Defaults to False.
        """
        super().__init__()
        self.feature_dim = 2048
        self.dilations = dilations
        self.neck_type = neck_type
        self.backbone: ResNetFeatureExtractor = self.build_backbone(
            backbone_name, frozen_backbone
        )
        self.neck = self.build_neck()
        self.head = Conv2dNormActivation(
            self.feature_dim // 8,
            output_size,
            1,
            norm_layer=None,
            activation_layer=None,
        )
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def build_backbone(
        self,
        backbone_name: Literal["resnet50", "resnet101", "resnet152"],
        frozen_backbone: bool,
    ) -> ResNetFeatureExtractor:
        """build backbone

        Args:
            backbone_name (Literal["resnet50", "resnet101", "resnet152"], optional): resnet network name.
            fronzen_backbone (bool, optional): froze the backbone or not.

        Returns:
            ResNetFeatureExtractor: feature extractor
        """
        backbone = ResNetFeatureExtractor(
            backbone_name,
            "block",
            trainable=not frozen_backbone,
        )
        feature_extractor = backbone.feature_extractor
        for key, layer in feature_extractor[3].named_modules():
            layer: nn.Conv2d
            if "conv2" in key:
                layer.dilation, layer.padding, layer.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in key:
                layer.stride = (1, 1)
        for key, layer in feature_extractor[4].named_modules():
            if "conv2" in key:
                layer.dilation, layer.padding, layer.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in key:
                layer.stride = (1, 1)

        backbone.forward = partial(backbone.forward, target_layer_names=["4_1"])

        return backbone

    def build_neck(self) -> nn.Module:
        """deeplab v3 use parallel ASPP and cascade ASPP

        Returns:
            nn.Module: neck module
        """
        if self.neck_type == "parallel":
            return ASPP(
                self.feature_dim,
                self.feature_dim // 8,
                self.feature_dim // 8,
                dilations=self.dilations,
            )
        elif self.neck_type == "cascade":
            return CascadeASPP(dilations=self.dilations)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.backbone(x)
        y = self.neck(features.pop())
        y = self.head(y)

        cropper = CenterCrop(x.shape[2:])
        y = self.upsample(y)
        y = cropper(y)
        return y
