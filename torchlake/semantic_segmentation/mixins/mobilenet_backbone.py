from typing import Literal

from torch import nn
from torchlake.common.models.mobilenet_feature_extractor import (
    MobileNetFeatureExtractor,
)


class MobileNetBackboneMixin:
    def build_backbone(
        self,
        backbone_name: Literal[
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
        ],
        frozen_backbone: bool,
    ) -> MobileNetFeatureExtractor:
        """build backbone, we use mobilenet

        Args:
            backbone_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"], optional): mobilenet network name.
            fronzen_backbone (bool, optional): froze the backbone or not.

        Returns:
            MobileNetFeatureExtractor: feature extractor
        """
        backbone = MobileNetFeatureExtractor(
            backbone_name,
            "block",
            trainable=not frozen_backbone,
        )
        feature_extractor = backbone.feature_extractor
        block_indices = backbone.get_stage()

        # please ignore coding redundancy
        for index in block_indices[-2]:
            for key, layer in feature_extractor[index].named_modules():
                layer: nn.Conv2d
                if "block.1.0" in key or "conv.1.0" in key:
                    kernel = layer.kernel_size[0]
                    dilation = 2
                    pad = (dilation * (kernel - 1)) // 2

                    layer.dilation, layer.padding = (2, 2), (pad, pad)

        penultimate_index = block_indices[-1]
        for index in penultimate_index:
            for key, layer in feature_extractor[index].named_modules():
                layer: nn.Conv2d
                if "block.1.0" in key or "conv.1.0" in key:
                    # o = i - d(k-1) + 2p
                    kernel = layer.kernel_size[0]
                    dilation = 4
                    pad = (dilation * (kernel - 1)) // 2

                    layer.dilation, layer.padding, layer.stride = (
                        (4, 4),
                        (pad, pad),  # +0 for v2, +2 for v3-S, +4 for v3-L
                        (1, 1),
                    )

        return backbone
