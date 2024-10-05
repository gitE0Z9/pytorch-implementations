from typing import Literal
import torch
from torch import nn
from torchlake.common.models import VGGFeatureExtractor


class NeuralStyleTransfer(nn.Module):
    def __init__(
        self,
        feature_extractor: VGGFeatureExtractor,
        content_layer_names: list[str],
        style_layer_names: list[str],
    ):
        super(NeuralStyleTransfer, self).__init__()
        self.content_layer_names = content_layer_names
        self.style_layer_names = style_layer_names

        self.feature_extractor = feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        type: Literal["content", "style"],
    ) -> list[torch.Tensor]:
        if type == "content":
            return self.feature_extractor(img, self.content_layer_names)
        elif type == "style":
            return self.feature_extractor(img, self.style_layer_names)
        else:
            raise NotImplementedError
