from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torchlake.common.models.feature_extractor_base import ExtractorBase

from .model import Extraction


class ExtractionFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        layer_type: Literal["block"],
        weight_path: Path | str | None = None,
        trainable: bool = True,
    ):
        """mobilenet feature extractor

        Args:
            network_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]): torchvision mobilenet model
            layer_type (Literal["block"]): extract which type of layer
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        self.weight_path = weight_path
        super().__init__("extraction", layer_type, trainable)

    def get_weight(self, _: str) -> str:
        return self.weight_path

    def build_feature_extractor(self, _: str, weight_path: str) -> Extraction:
        feature_extractor = Extraction(output_size=1000)

        if weight_path is not None:
            feature_extractor.load_state_dict(torch.load(weight_path))

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        del feature_extractor.head[-1]

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: Literal["0_1", "1_1", "2_1", "output"],
    ) -> list[torch.Tensor]:
        if self.layer_type != "block":
            raise NotImplementedError

        features = []

        y = img
        y = self.feature_extractor.foot(y)
        if "0_1" in target_layer_names:
            features.append(y)

        stage_count = 0
        for layer in self.feature_extractor.blocks:
            y = layer(y)
            if isinstance(layer, nn.MaxPool2d):
                stage_count += 1
                if f"{stage_count}_1" in target_layer_names:
                    features.append(y)

        if "output" in target_layer_names:
            y = self.feature_extractor.head(y)
            features.append(y)

        return features
