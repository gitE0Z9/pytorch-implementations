from typing import Literal
from torch import nn
from torchlake.common.models import VGGFeatureExtractor


class DeepLabStyleVGGBackboneMixin:
    def build_backbone(
        self,
        backbone_name: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
        frozen_backbone: bool,
    ) -> VGGFeatureExtractor:
        """build backbone

        Args:
            fronzen_backbone (bool, optional): froze the backbone or not. Defaults to False.

        Returns:
            VGGFeatureExtractor: feature extractor
        """
        backbone = VGGFeatureExtractor(
            backbone_name,
            "maxpool",
            trainable=not frozen_backbone,
        )
        feature_layers = backbone.feature_extractor
        # stage 5 convs
        for i in range(1, 4):
            conv_layer: nn.Conv2d = feature_layers[-1 - i * 2]
            conv_layer.dilation = (2, 2)
            conv_layer.padding = (2, 2)

        # skip subsampling and keep 8x
        stage = 0
        for layer in feature_layers:
            if isinstance(layer, nn.MaxPool2d):
                stage += 1
                layer.padding = (1, 1)
                layer.kernel_size = (3, 3)

                # stage 4, 5
                if stage >= 4:
                    layer.stride = (1, 1)

        return backbone
