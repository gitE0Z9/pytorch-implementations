from typing import Literal
from torch import nn
from torchlake.common.models import ResNetFeatureExtractor


class DeepLabStyleResNetBackboneMixin:
    def build_backbone(
        self,
        backbone_name: Literal["resnet50", "resnet101", "resnet152"],
        frozen_backbone: bool,
    ) -> ResNetFeatureExtractor:
        """build backbone

        Args:
            backbone_name (Literal["resnet50", "resnet101", "resnet152"], optional): resnet network name.
            fronzen_backbone (bool, optional): froze the backbone or not.

        Returns:
            ResNetFeatureExtractor: feature extractor
        """
        backbone = ResNetFeatureExtractor(
            backbone_name,
            "block",
            trainable=not frozen_backbone,
        )
        feature_extractor = backbone.feature_extractor
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
