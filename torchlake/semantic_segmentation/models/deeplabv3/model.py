from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import ASPP, CascadeASPP


class DeepLabV3(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        hidden_dim: int = 256,
        output_size: int = 1,
        dilations: Sequence[int] = (6, 12, 18),
        neck_type: Literal["parallel", "cascade"] = "parallel",
    ):
        """DeepLab v3 in paper [1706.05587v3]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            dilations (list[int], optional):
                dilation sizes of ASPP, for 16x (6, 12, 18), for 8x (12, 24, 36),
                as for cascade ASPP it should be (8, 16, 1) for 16x and (16, 32, 1) for 8x. Defaults to (6, 12, 18).
            neck_type: (Literal["parallel", "cascade"], optional): neck type is multi-grid cascade module or parallel ASPP. Defaults to "parallel",
        """
        self.hidden_dim = hidden_dim
        self.dilations = dilations
        self.neck_type = neck_type
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("4_1",))

    def build_neck(self) -> nn.Module:
        """deeplab v3 use parallel ASPP and cascade ASPP

        Returns:
            nn.Module: neck module
        """
        self.neck = nn.Sequential()

        if self.neck_type == "parallel":
            self.neck.append(
                ASPP(
                    self.foot.hidden_dim_32x,
                    hidden_dim=self.hidden_dim,
                    dilations=self.dilations,
                )
            )
        elif self.neck_type == "cascade":
            self.neck.append(
                CascadeASPP(
                    self.foot.hidden_dim_32x,
                    hidden_dim=self.hidden_dim,
                    output_channel=self.foot.hidden_dim_32x,
                    dilations=self.dilations,
                )
            )
        else:
            raise NotImplementedError

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.foot(x)
        y = self.neck(features.pop())
        y = self.head(y)

        cropper = CenterCrop(x.shape[2:])
        y = F.interpolate(y, scale_factor=8, mode="bilinear", align_corners=True)
        y = cropper(y)
        return y
