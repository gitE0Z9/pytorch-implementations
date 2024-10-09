from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)

from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class MobileNetFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: Literal[
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
        ],
        layer_type: Literal["block"],
        trainable: bool = True,
    ):
        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    def get_weight(self, network_name: str) -> Weights:
        return {
            "mobilenet_v2": MobileNet_V2_Weights.DEFAULT,
            "mobilenet_v3_small": MobileNet_V3_Small_Weights.DEFAULT,
            "mobilenet_v3_large": MobileNet_V3_Large_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, network_name)
        feature_extractor: nn.Module = model_class(weights=weights).features

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: list[str],
    ) -> list[torch.Tensor]:
        if self.layer_type != "block":
            raise NotImplementedError

        features = []

        img = self.normalization(img)

        y = img
        block_count = -1
        layer_count = 0
        for layer in self.feature_extractor:
            layer_count += 1

            input_shape = y.shape
            y: torch.Tensor = layer(y)
            output_shape = y.shape

            if input_shape[2:] != output_shape[2:]:
                block_count += 1
                layer_count = 1

            layer_name = f"{block_count}_{layer_count}"
            if layer_name in target_layer_names:
                features.append(y)

        if "output" in target_layer_names:
            features.append(y.mean((2, 3)))

        return features
