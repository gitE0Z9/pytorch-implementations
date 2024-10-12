from typing import Literal

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from ...mixins.vgg_backbone import MSCADStyleVGGBackboneMixin
from .network import ContextModule, context_network_basic, context_network_large


class MSCAD(MSCADStyleVGGBackboneMixin, nn.Module):

    def __init__(
        self,
        output_size: int,
        context_type: Literal["basic", "large"] = "large",
        frozen_backbone: bool = True,
    ):
        super().__init__()
        fc_dim = 4096
        self.context_type = context_type

        self.backbone = self.build_backbone("vgg16", frozen_backbone)
        self.convs = nn.Sequential(
            Conv2dNormActivation(
                512,
                fc_dim,
                7,
                dilation=4,
                norm_layer=None,
            ),
            nn.Dropout(p=0.5),
            Conv2dNormActivation(fc_dim, fc_dim, 1, norm_layer=None),
            nn.Dropout(p=0.5),
            Conv2dNormActivation(fc_dim, output_size, 1, norm_layer=None),
        )
        self.neck = self.get_neck()(output_size)
        self.head = nn.ConvTranspose2d(
            output_size,
            output_size,
            16,
            stride=8,
            padding=4,
            groups=output_size,
            bias=False,
        )

    def get_neck(self) -> ContextModule:
        return {
            "basic": context_network_basic,
            "large": context_network_large,
        }[self.context_type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward(x, ["5_1"])
        y = self.convs(features.pop())
        y = self.neck(y)
        return self.head(y)
