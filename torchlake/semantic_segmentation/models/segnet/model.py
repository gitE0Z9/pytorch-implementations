import torch
from torch import nn

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import DecoderBlock


class SegNet(ModelBase):
    def __init__(self, backbone: ExtractorBase, output_size: int = 1):
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _: int, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                DecoderBlock(self.foot.hidden_dim_32x, self.foot.hidden_dim_16x, 3),
                DecoderBlock(self.foot.hidden_dim_16x, self.foot.hidden_dim_8x, 3),
                DecoderBlock(self.foot.hidden_dim_8x, self.foot.hidden_dim_4x, 3),
                DecoderBlock(self.foot.hidden_dim_4x, self.foot.hidden_dim_2x, 2),
                DecoderBlock(self.foot.hidden_dim_2x, self.foot.hidden_dim_2x, 2),
            ]
        )

    def build_head(self, output_size: int, **kwargs):
        self.head = nn.Sequential(
            nn.Conv2d(self.foot.hidden_dim_2x, output_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, pooling_indices = self.foot(x)

        y = features.pop()
        for neck in self.neck:
            y = neck(y, pooling_indices.pop())

        return self.head(y)
