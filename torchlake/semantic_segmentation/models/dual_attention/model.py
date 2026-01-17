from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import DualAttention2d


class DANet(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        reduction_ratio: float = 32,
        dropout_prob: float = 0.1,
    ):
        """Dual Attention Network for Scene Segmentation [1809.02983v4]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            reduction_ratio (float, optional): _description_. Defaults to 32.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.reduction_ratio = reduction_ratio
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            DualAttention2d(self.foot.hidden_dim_32x, self.reduction_ratio),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            nn.Dropout2d(self.dropout_prob),
            nn.Conv2d(self.foot.hidden_dim_32x // self.reduction_ratio, output_size, 1),
        )

    def train(self: DANet, mode: bool = True) -> DANet:
        result = super().train(mode)

        if not hasattr(self, "aux"):
            self.aux = nn.Sequential(
                Conv2dNormActivation(
                    self.foot.hidden_dim_16x,
                    self.foot.hidden_dim_16x // 4,
                    3,
                ),
                nn.Dropout2d(p=self.dropout_prob),
                nn.Conv2d(self.foot.hidden_dim_16x // 4, self.output_size, 1),
            )

        self.foot.fix_target_layers(("3_1", "4_1"))

        return result

    def eval(self: DANet) -> DANet:
        result = super().eval()

        self.foot.fix_target_layers(("4_1",))

        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor] | torch.Tensor:
        features: list[torch.Tensor] = self.foot(x)

        if self.training:
            aux, y = features
        else:
            y = features.pop()

        y = self.neck(y)
        y = self.head(y)
        y = F.interpolate(y, x.shape[2:], mode="bilinear")

        if self.training:
            aux = self.aux(aux)
            aux = F.interpolate(aux, x.shape[2:], mode="bilinear")
            return y, aux

        return y
