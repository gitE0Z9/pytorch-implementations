from functools import partial
from typing import Literal

import torch
from torch import nn
from torchlake.common.models import MobileNetFeatureExtractor
from torchvision.transforms import CenterCrop

from ...mixins.mobilenet_backbone import MobileNetBackboneMixin
from ..deeplabv3.network import ASPP


class MobileNetV2Seg(MobileNetBackboneMixin, nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        reduction_ratio: int = 8,
        backbone_name: Literal[
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
        ] = "mobilenet_v2",
        frozen_backbone: bool = False,
    ):
        """MobileNet v2 semantic segmentation in paper [1801.04381v4]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            reduction_ratio (int, optional): prediction head dimension reduced ratio. Defaults to 8.
            backbone_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"], optional): mobilenet network name. Defaults to "mobilenet_v2".
            fronzen_backbone (bool, optional): froze the backbone or not. Defaults to False.
        """
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.backbone: MobileNetFeatureExtractor = self.build_backbone(
            backbone_name, frozen_backbone
        )
        self.backbone.forward = partial(
            self.backbone.forward,
            target_layer_names=[f"4_{len(self.backbone.get_stage()[-1])-1}"],
        )
        self.feature_dim = self.backbone.get_feature_dim()[-1][-2]
        self.neck = self.build_neck()
        self.head = nn.Conv2d(self.feature_dim // self.reduction_ratio, output_size, 1)
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def build_neck(self) -> nn.Module:
        """ASPP withotut 3x3

        Returns:
            nn.Module: neck module
        """
        return ASPP(
            self.feature_dim,
            self.feature_dim // self.reduction_ratio,
            self.feature_dim // self.reduction_ratio,
            dilations=[],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.backbone(x)
        y = self.neck(features.pop())
        y = self.head(y)

        cropper = CenterCrop(x.shape[2:])
        y = self.upsample(y)
        y = cropper(y)
        return y
