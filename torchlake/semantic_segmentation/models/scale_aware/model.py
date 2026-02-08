from typing import Sequence

import torch
import torch.nn.functional as F

from torchlake.common.models.model_base import ModelBase
from .network import ScaleAwareAttention


class ScaleAware(ModelBase):

    def __init__(
        self,
        model: ModelBase,
        scales: Sequence[int],
        hidden_dim: int = 512,
    ):
        self.scales = scales
        self.hidden_dim = hidden_dim
        super().__init__(
            model.input_channel,
            model.output_size,
            foot_kwargs={"model": model},
        )

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("model")

    def build_head(self, output_size, **kwargs):
        self.head = ScaleAwareAttention(
            output_size * len(self.scales),
            hidden_dim=self.hidden_dim,
            num_scales=len(self.scales),
        )

    def forward(self, x: torch.Tensor, output_attention: bool = False) -> torch.Tensor:
        outputs = []
        for scale in self.scales:
            y: torch.Tensor = F.interpolate(
                x,
                scale_factor=scale,
                mode="bilinear",
                align_corners=True,
            )
            y = self.foot(y)
            y = F.interpolate(
                y,
                scale_factor=1 / scale,
                mode="bilinear",
                align_corners=True,
            )
            outputs.append(y)

        # B, S, H, W
        a = self.head(outputs)
        # B, 1, C, H, W x B, S, 1, H, W => B, S, C, H, W => B, C, H, W
        y = (y.unsqueeze(1) * a.unsqueeze(2)).sum(1)

        if self.training:
            outputs.append(y)

            if output_attention:
                return outputs, a
            else:
                return outputs

        if output_attention:
            return y, a
        else:
            return y
