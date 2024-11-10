from functools import partial
from typing import Literal

import torch
from torch import nn
from torchlake.common.models import MobileNetFeatureExtractor
from torchvision.transforms import CenterCrop

from ...mixins.mobilenet_backbone import MobileNetBackboneMixin
from .network import LRASPP


class MobileNetV3Seg(MobileNetBackboneMixin, nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        hidden_dim: int = 128,
        pool_kernel_size: tuple[int] = (49, 49),
        pool_stride: tuple[int] = (16, 20),
        backbone_name: Literal[
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
        ] = "mobilenet_v3_large",
        frozen_backbone: bool = False,
    ):
        """MobileNet v3 semantic segmentation in paper [1905.02244v5]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            hidden_dim (int): dimension of lr-aspp layer
            pool_kernel_size (tuple[int], optional): kernel size of pool. Defaults to (49, 49).
            pool_stride (tuple[int], optional): stride of pool. Defaults to (16, 20).
            backbone_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"], optional): mobilenet network name. Defaults to "mobilenet_v3_large".
            fronzen_backbone (bool, optional): froze the backbone or not. Defaults to False.
        """
        super().__init__()
        self.backbone: MobileNetFeatureExtractor = self.build_backbone(
            backbone_name, frozen_backbone
        )
        self.backbone.forward = partial(
            self.backbone.forward,
            target_layer_names=[
                f"2_{len(self.backbone.get_stage()[-3])}",
                f"4_{len(self.backbone.get_stage()[-1])-1}",
            ],
        )
        self.feature_dims = [
            self.backbone.feature_dims[-3][-1],
            self.backbone.feature_dims[-1][-2],
        ]
        self.head: LRASPP = self.build_head(
            hidden_dim,
            output_size,
            pool_kernel_size,
            pool_stride,
        )
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def build_head(
        self,
        hidden_dim: int,
        output_size: int,
        pool_kernel_size: tuple[int] = (49, 49),
        pool_stride: tuple[int] = (16, 20),
    ) -> LRASPP:
        """lr aspp

        Args:
            hidden_dim (int): dimension of intermediate layer
            output_size (int, optional): output size.
            pool_kernel_size (tuple[int], optional): kernel size of pool. Defaults to (49, 49).
            pool_stride (tuple[int], optional): stride of pool. Defaults to (16, 20).
        Returns:
            LRASPP: lr aspp
        """
        return LRASPP(
            self.feature_dims,
            hidden_dim,
            output_size,
            pool_kernel_size,
            pool_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.backbone(x)
        y = self.head.forward(*features)

        cropper = CenterCrop(x.shape[2:])
        y = self.upsample(y)
        y = cropper(y)
        return y
