# for further speed up, split import is promising
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from ...mixins.vgg_backbone import DeepLabStyleVGGBackboneMixin
from .network import GlobalContextModule


class ParseNet(DeepLabStyleVGGBackboneMixin, nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        frozen_backbone: bool = False,
    ):
        """ParseNet [1506.04579v2]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            fronzen_backbone (bool, optional): froze the vgg backbone or not. Defaults to False.
        """
        super().__init__()
        self.backbone = self.build_backbone("vgg16", frozen_backbone)
        self.neck = nn.Sequential(
            Conv2dNormActivation(
                512,
                1024,
                3,
                padding=12,
                dilation=12,
                norm_layer=None,
            ),
            nn.Dropout(0.5),
            Conv2dNormActivation(1024, 1024, 1, norm_layer=None),
            nn.Dropout(0.5),
            GlobalContextModule(1024, output_size),
        )
        self.head = nn.ConvTranspose2d(
            output_size,
            output_size,
            16,
            stride=8,
            padding=4,
            groups=output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward(x, ["4_1"])
        y = self.neck(features.pop())
        return self.head(y)
