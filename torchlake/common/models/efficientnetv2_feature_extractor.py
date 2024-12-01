from torchvision.models.efficientnet import (
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)


from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights

from ..types import EFFICIENTNET_V2_NAMES
from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class EfficientNetV2FeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: EFFICIENTNET_V2_NAMES,
        layer_type: Literal["block"],
        trainable: bool = True,
    ):
        """resnet feature extractor

        Args:
            network_name (EFFICIENTNET_V2_NAMES): torchvision efficientnet v2 model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super().__init__(network_name, layer_type, trainable)
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        return {
            "s": [24, 24, 48, 64, 128, 160, 256, 1280],
            "m": [24, 24, 48, 80, 160, 176, 304, 512, 1280],
            "l": [32, 32, 64, 96, 192, 224, 384, 640, 1280],
        }[self.network_name]

    def get_weight(self, network_name: str) -> Weights:
        return {
            "s": EfficientNet_V2_S_Weights.DEFAULT,
            "m": EfficientNet_V2_M_Weights.DEFAULT,
            "l": EfficientNet_V2_L_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(torchvision.models, f"efficientnet_v2_{network_name}")
        feature_extractor: nn.Module = model_class(weights=weights).features

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: Literal[
            "0_1",
            "1_1",
            "2_1",
            "3_1",
            "4_1",
            "5_1",
            "6_1",
            "7_1",
            "8_1",
            "output",
        ],
    ) -> list[torch.Tensor]:
        if self.layer_type != "block":
            raise NotImplementedError

        features = []

        img = self.normalization(img)

        y = img
        for i, layer in enumerate(self.feature_extractor):
            y = layer(y)
            if f"{i}_1" in target_layer_names:
                features.append(y)

        if "output" in target_layer_names:
            features.append(y.mean((2, 3)).squeeze(-1))

        return features
