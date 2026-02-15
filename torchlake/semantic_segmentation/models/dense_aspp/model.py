from typing import Sequence

import torch
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import DenseASPP


class DeepLabV3WithDenseASPP(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 256,
        dilations: Sequence[int] = (3, 6, 12, 18, 24),
        output_stride: int = 8,
    ):
        """DeepLab v3 with DenseASPP in the paper [1706.05587v3]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            hidden_dim (int, optional): hidden dimension of ASPP. Defaults to 256.
            dilations (list[int], optional): dilation sizes of ASPP, (3, 6, 12, 18, 24). Defaults to (3, 6, 12, 18, 24).
            output_stride (int, optional): downscale ratio of the feature map to the input, Defaults to 8.
        """
        self.hidden_dim = hidden_dim
        self.output_stride = output_stride
        self.dilations = dilations
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self) -> nn.Module:
        self.neck = nn.Sequential(
            DenseASPP(
                self.foot.hidden_dim_32x,
                hidden_dim=self.hidden_dim,
                dilations=self.dilations,
            )
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(
                self.foot.hidden_dim_32x + len(self.dilations) * self.hidden_dim,
                output_size,
                1,
            ),
            # nn.Conv2d(self.hidden_dim, output_size, 1),
            nn.Upsample(
                scale_factor=self.output_stride,
                mode="bilinear",
                align_corners=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = self.foot(x)
        y = self.neck(features.pop())

        cropper = CenterCrop(x.shape[2:])
        return cropper(self.head(y))
