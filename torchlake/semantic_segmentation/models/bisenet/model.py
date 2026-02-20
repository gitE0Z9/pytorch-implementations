import torch
from torch import nn
import torch.nn.functional as F
from torchlake.common.models.model_base import ModelBase
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop

from .network import FeatureFusionModule, AttentionRefinementModule


class BiSeNet(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 64,
    ):
        self.hidden_dim = hidden_dim
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _: int, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleDict(
            {
                "spatial_path": nn.Sequential(
                    Conv2dNormActivation(
                        self.input_channel, self.hidden_dim, 3, stride=2
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim, self.hidden_dim * 2, 3, stride=2
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim * 2, self.hidden_dim * 4, 3, stride=2
                    ),
                ),
                "context_path": nn.ModuleList(
                    # from high to low
                    [
                        AttentionRefinementModule(self.foot.hidden_dim_32x),
                        AttentionRefinementModule(self.foot.hidden_dim_16x),
                    ]
                ),
            }
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            # from high to low
            [
                nn.Identity(),
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    ),
                    nn.Conv2d(
                        self.foot.hidden_dim_32x,
                        self.foot.hidden_dim_16x,
                        1,
                    ),
                ),
            ]
        )

    def build_head(self, output_size: int, **kwargs):
        h = self.hidden_dim * 4 + self.foot.hidden_dim_16x
        self.head = nn.ModuleList(
            [
                FeatureFusionModule(h),
                nn.Conv2d(h, output_size, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = self.foot(x)

        y = features.pop()[:, :, None, None]
        for block, neck in zip(self.blocks["context_path"], self.neck):
            y = neck(y)
            z: torch.Tensor = block(features.pop())
            cropper = CenterCrop(z.shape[2:])
            y = cropper(y) + z

        y = self.head[0](self.blocks["spatial_path"](x), y)
        y = self.head[1](y)

        if not self.training:
            y = F.interpolate(
                y,
                scale_factor=8,
                mode="bilinear",
                align_corners=True,
            )
            cropper = CenterCrop(x.shape[2:])
            y = cropper(y)

        return y
