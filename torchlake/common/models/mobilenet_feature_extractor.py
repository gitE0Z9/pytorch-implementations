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
        """mobilenet feature extractor

        Args:
            network_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]): torchvision mobilenet model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    def get_feature_dim(self) -> int:
        return {
            "mobilenet_v2": [
                [32, 16],
                [24] * 2,
                [32] * 3,
                [64] * 4 + [96] * 3,
                [160] * 3 + [320] + [1280],
            ],
            "mobilenet_v3_small": [
                [16],
                [16],
                [24],
                [40] * 3 + [48] * 2,
                [96] * 3 + [576],
            ],
            "mobilenet_v3_large": [
                [16, 16],
                [24] * 2,
                [40] * 3,
                [80] * 4 + [112] * 2,
                [160] * 3 + [960],
            ],
        }[self.network_name]

    def get_stage(self) -> list[list[int]]:
        return {
            "mobilenet_v2": [
                [0, 1],
                [2, 3],
                [4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18],
            ],
            "mobilenet_v3_small": [
                [0],
                [1],
                [2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            "mobilenet_v3_large": [
                [0, 1],
                [2, 3],
                [4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
        }[self.network_name]

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
        for stage_idx, stage in enumerate(self.get_stage()):
            for layer_count, layer_idx in enumerate(stage):
                layer = self.feature_extractor[layer_idx]
                y: torch.Tensor = layer(y)

                layer_name = f"{stage_idx}_{layer_count+1}"
                if layer_name in target_layer_names:
                    features.append(y)

        if "output" in target_layer_names:
            features.append(y.mean((2, 3)))

        return features
