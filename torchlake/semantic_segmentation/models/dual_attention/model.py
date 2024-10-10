from typing import Literal

import torch
import torch.nn.functional as F
from annotated_types import T
from torch import nn
from torchvision.ops import Conv2dNormActivation

from ...mixins.resnet_backbone import DeepLabStyleResNetBackboneMixin
from .network import DualAttention2d


class DANet(DeepLabStyleResNetBackboneMixin, nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        reduction_ratio: float = 32,
        dropout_prob: float = 0.1,
        backbone_name: Literal["resnet50", "resnet101", "resnet152"] = "resnet50",
        frozen_backbone: bool = False,
    ):
        """Dual Attention Network for Scene Segmentation [1809.02983v4]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            reduction_ratio (float, optional): _description_. Defaults to 32.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            backbone_name (Literal["resnet50", "resnet101", "resnet152"], optional): resnet network name. Defaults to "resnet50".
            fronzen_backbone (bool, optional): froze the resnet backbone or not. Defaults to False.
        """
        super().__init__()
        hidden_dim = 2048
        self.dropout_prob = dropout_prob
        self.output_size = output_size
        self.backbone = self.build_backbone(backbone_name, frozen_backbone)
        self.neck = DualAttention2d(hidden_dim, reduction_ratio)
        self.head = nn.Sequential(
            nn.Dropout2d(dropout_prob, False),
            nn.Conv2d(hidden_dim // reduction_ratio, output_size, 1),
        )

    def train(self: T, mode: bool = True) -> T:
        result = super().train(mode)

        if not hasattr(self, "aux"):
            hidden_dim = 1024
            self.aux = nn.Sequential(
                Conv2dNormActivation(hidden_dim, hidden_dim // 4),
                nn.Dropout2d(p=self.dropout_prob),
                nn.Conv2d(hidden_dim // 4, self.output_size, 1),
            )

        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor] | torch.Tensor:
        # extract features
        feature_names = ["4_1"]
        if self.training:
            feature_names.append("3_1")
        features: list[torch.Tensor] = self.backbone(x, feature_names)

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
