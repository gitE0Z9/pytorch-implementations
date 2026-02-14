import math

import torch
from torch import nn
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import ChannelAttentionBlock, RefinementResidualBlock


class DFN(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 256,
        output_stride: int = 4,
    ):
        self.hidden_dim = hidden_dim
        self.output_stride = output_stride
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    @property
    def num_skip_connection(self) -> int:
        return 5 - int(math.log2(self.output_stride))

    def build_foot(self, input_channel, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList()

        s = 32
        while self.output_stride <= s:
            self.blocks.append(
                RefinementResidualBlock(
                    getattr(self.foot, f"hidden_dim_{s}x"),
                    self.hidden_dim,
                ),
            )
            s //= 2

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList([])

        s = 32
        while self.output_stride <= s:
            self.neck.append(
                nn.Sequential(
                    ChannelAttentionBlock(
                        self.hidden_dim,
                        self.hidden_dim if s != 32 else self.foot.hidden_dim_32x,
                        self.hidden_dim,
                    ),
                    RefinementResidualBlock(self.hidden_dim, self.hidden_dim),
                )
            )
            s //= 2

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, output_size, 1),
            nn.Upsample(
                scale_factor=self.output_stride,
                mode="bilinear",
                align_corners=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = self.foot(x)
        # global feature
        y = features.pop()[:, :, None, None]

        for block, neck in zip(self.blocks, self.neck):
            y = neck[0](block(features.pop()), y)
            y = neck[1](y)

        y = self.head(y)
        cropper = CenterCrop(x.shape[2:])
        return cropper(y)
