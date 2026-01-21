from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from ..deeplabv3.network import ASPP
from .network import Decoder


class DeepLabV3Plus(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        hidden_dim: int = 256,
        shallow_hidden_dim: int = 48,
        output_size: int = 1,
        dilations: Sequence[int] = (6, 12, 18),
    ):
        """DeepLab v3+ in paper [1802.02611v3]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            dilations (Sequence[int], optional): dilation size of ASPP, for 16x [6, 12, 18], for 8x [12, 24, 36]. Defaults to [6, 12, 18].
        """
        self.hidden_dim = hidden_dim
        self.shallow_hidden_dim = shallow_hidden_dim
        self.dilations = dilations
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("1_1", "4_1"))

    def build_neck(self) -> nn.Module:
        """deeplab v3 use parallel ASPP

        Returns:
            nn.Module: neck module
        """
        self.neck = nn.Sequential(
            ASPP(self.foot.hidden_dim_32x, self.hidden_dim, dilations=self.dilations),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            Decoder(
                self.foot.hidden_dim_4x,
                self.hidden_dim,
                self.shallow_hidden_dim,
                upsample_scale=2,
                hidden_dim=self.hidden_dim,
                output_channel=output_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8x
        features: list[torch.Tensor] = self.foot(x)
        y = self.neck(features.pop())

        y = self.head[0](features.pop(), y)

        cropper = CenterCrop(x.shape[2:])
        y = F.interpolate(y, scale_factor=4, mode="bilinear", align_corners=True)
        y = cropper(y)
        return y
