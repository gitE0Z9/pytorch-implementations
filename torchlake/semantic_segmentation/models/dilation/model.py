from typing import Literal

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
import torch.nn.functional as F

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import context_module_basic, context_module_large
from ..fcn.network import init_deconv_with_bilinear_kernel


class DilationNet(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        context_type: Literal["basic", "large"] | None = "basic",
    ):
        self.context_type = context_type
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("6_1",))

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential(
            nn.Conv2d(self.foot.feature_dim, self.output_size, 1),
        )

        nn.init.normal_(self.blocks[0].weight.data, 0, 1e-3)
        nn.init.zeros_(self.blocks[0].bias.data)

    def build_neck(self, **kwargs):
        if self.context_type is None:
            layer = nn.Identity()
        else:
            module = {
                "basic": context_module_basic,
                "large": context_module_large,
            }[self.context_type]
            layer = module(self.output_size)

        self.neck = nn.Sequential(
            layer,
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.ConvTranspose2d(
                output_size,
                output_size,
                16,
                stride=8,
                padding=4,
                groups=output_size,
                bias=False,
            )
        )

        init_deconv_with_bilinear_kernel(self.head[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.foot(x)
        y = self.blocks(features.pop())
        y = self.neck(y)
        y = self.head(y)

        return y
