# for further speed up, split import is promising
import torch
from torch import nn

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from ..fcn.network import init_deconv_with_bilinear_kernel
from .network import GlobalContextModule


class ParseNet(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        scale: float = 10.0,
    ):
        """ParseNet [1506.04579v2]

        Args:
            backbone (ExtractorBase): backbone.
            output_size (int, optional): output size. Defaults to 1.
            scale (float, optional): L2 norm initial scale. Defaults to 10.0.
        """
        self.scale = scale
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("6_1"))

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            GlobalContextModule(
                self.foot.feature_dim,
                self.output_size,
                scale=self.scale,
            ),
        )

    def build_head(self, output_size, **kwargs):
        layer = nn.ConvTranspose2d(
            output_size,
            output_size,
            16,
            stride=8,
            padding=4,
            groups=output_size,
            bias=False,
        )
        init_deconv_with_bilinear_kernel(layer)

        self.head = nn.Sequential(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.foot(x)
        y = self.neck(features.pop())
        return self.head(y)
