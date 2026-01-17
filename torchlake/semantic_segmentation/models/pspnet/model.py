from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import PyramidPool2d


class PSPNet(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        bins_size: list[int] = [1, 2, 3, 6],
        dropout_prob: float = 0.5,
    ):
        """Pyramid spatial pooling network [1612.01105v2]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            bins_size (list[int], optional): size of pooled feature maps. Defaults to [1, 2, 3, 6].
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.bins_size = bins_size
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        self.neck = nn.Sequential(
            PyramidPool2d(self.foot.hidden_dim_32x, self.bins_size),
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            Conv2dNormActivation(
                self.foot.hidden_dim_32x * 2,
                self.foot.hidden_dim_32x // 4,
                3,
            ),
            nn.Dropout2d(self.dropout_prob),
            nn.Conv2d(
                self.foot.hidden_dim_32x // 4,
                output_size,
                1,
            ),
        )

    def train(self: PSPNet, mode: bool = True) -> PSPNet:
        result = super().train(mode)

        if not hasattr(self, "aux"):
            self.aux = nn.Sequential(
                Conv2dNormActivation(
                    self.foot.hidden_dim_16x,
                    self.foot.hidden_dim_16x // 4,
                    3,
                ),
                nn.Dropout2d(p=self.dropout_prob),
                nn.Conv2d(
                    self.foot.hidden_dim_16x // 4,
                    self.output_size,
                    1,
                ),
            )

        self.foot.fix_target_layers(("3_1", "4_1"))

        return result

    def eval(self: PSPNet) -> PSPNet:
        result = super().eval()

        self.foot.fix_target_layers(("4_1",))

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = self.foot(x)

        if self.training:
            aux, y = features
        else:
            y = features.pop()

        # head
        y = self.neck(y)
        y = self.head(y)
        y = F.interpolate(y, x.shape[2:], mode="bilinear")

        if self.training:
            aux = self.aux(aux)
            aux = F.interpolate(aux, x.shape[2:], mode="bilinear")
            return y, aux

        return y
