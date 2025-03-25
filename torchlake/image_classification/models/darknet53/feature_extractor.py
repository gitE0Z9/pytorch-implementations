from pathlib import Path
from typing import Literal

import torch
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchvision.ops import Conv2dNormActivation
from .model import DarkNet53


class DarkNet53FeatureExtractor(ExtractorBase):
    def __init__(
        self,
        layer_type: Literal["block"],
        weight_path: Path | str | None = None,
        trainable: bool = True,
    ):
        """darknet53 feature extractor

        Args:
            layer_type (Literal["block"]): extract which type of layer
            weight_path (Path|str|None, optional): path to pytorch weight file. Defaults to None.
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        self.weight_path = weight_path
        super().__init__("darknet53", layer_type, trainable)

    @property
    def feature_dims(self) -> list[int]:
        return [64, 128, 256, 512, 1024]

    def get_weight(self, _: str) -> str:
        return self.weight_path

    def build_feature_extractor(self, _: str, weight_path: str) -> DarkNet53:
        feature_extractor = DarkNet53(output_size=1000)

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
        target_layer_names: Literal["0_1", "1_1", "2_1", "3_1", "4_1", "output"],
    ) -> list[torch.Tensor]:
        if self.layer_type not in ["block"]:
            raise NotImplementedError

        features = []

        y = img

        y = self.feature_extractor.foot(y)

        stage_count = 0
        for layer in self.feature_extractor.blocks:
            extract_cond = (
                isinstance(layer, Conv2dNormActivation)
                and f"{stage_count}_1" in target_layer_names
            )

            y = layer(y)

            if self.layer_type == "block" and extract_cond:
                features.append(y)

            if isinstance(layer, Conv2dNormActivation):
                stage_count += 1

        if "output" in target_layer_names:
            y = self.feature_extractor.head(y)
            features.append(y)

        return features
