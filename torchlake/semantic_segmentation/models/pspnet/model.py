from typing import Literal

import torch
import torch.nn.functional as F
from annotated_types import T
from torch import nn
from torchlake.common.models import ResNetFeatureExtractor
from torchvision.ops import Conv2dNormActivation

from .network import PyramidPool2d


class PSPNet(nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        bins_size: list[int] = [1, 2, 3, 6],
        dropout_prob: float = 0.5,
        backbone_name: Literal["resnet50", "resnet101", "resnet152"] = "resnet50",
        frozen_backbone: bool = False,
    ):
        """Pyramid spatial pooling network [1612.01105v2]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            bins_size (list[int], optional): size of pooled feature maps. Defaults to [1, 2, 3, 6].
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            backbone_name (Literal["resnet50", "resnet101", "resnet152"], optional): resnet network name. Defaults to "resnet50".
            fronzen_backbone (bool, optional): froze the resnet backbone or not. Defaults to False.
        """
        super().__init__()
        hidden_dim = 2048
        self.dropout_prob = dropout_prob
        self.output_size = output_size
        self.backbone = self.build_backbone(backbone_name, frozen_backbone)
        self.neck = PyramidPool2d(hidden_dim, bins_size)
        self.head = nn.Sequential(
            Conv2dNormActivation(hidden_dim * 2, hidden_dim // 4, 3),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(hidden_dim // 4, output_size, 1),
        )

    def build_backbone(
        self,
        backbone_name: Literal["resnet50", "resnet101", "resnet152"] = "resnet50",
        frozen_backbone: bool = False,
    ) -> ResNetFeatureExtractor:
        """build resnet backbone of PSPNet

        Args:
            backbone_name (Literal["resnet50", "resnet101", "resnet152"], optional): resnet network name. Defaults to "resnet50".
            fronzen_backbone (bool, optional): froze the resnet backbone or not. Defaults to False.

        Returns:
            ResNetFeatureExtractor: feature extractor
        """
        backbone = ResNetFeatureExtractor(
            backbone_name,
            "block",
            trainable=not frozen_backbone,
        )

        feature_extractor = backbone.feature_extractor

        # deeplab v2 style
        # memory hungry !!!
        # https://github.com/hszhao/semseg/blob/4f274c3f276778228bc14a4565822d46359f0cc8/model/pspnet.py#L49
        for key, layer in feature_extractor[3].named_modules():
            layer: nn.Conv2d
            if "conv2" in key:
                layer.dilation, layer.padding, layer.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in key:
                layer.stride = (1, 1)
        for key, layer in feature_extractor[4].named_modules():
            if "conv2" in key:
                layer.dilation, layer.padding, layer.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in key:
                layer.stride = (1, 1)

        return backbone

    def train(self: T, mode: bool = True) -> T:
        result = super().train(mode)

        if not hasattr(self, "aux"):
            hidden_dim = 1024
            self.aux = nn.Sequential(
                Conv2dNormActivation(hidden_dim, hidden_dim // 4, 3),
                nn.Dropout2d(p=self.dropout_prob),
                nn.Conv2d(hidden_dim // 4, self.output_size, 1),
            )

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract features
        feature_names = ["4_1"]
        if self.training:
            feature_names.append("3_1")
        features: list[torch.Tensor] = self.backbone(x, feature_names)

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
