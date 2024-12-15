import torch
import torchvision
from torch import nn
from torchvision.models._api import Weights
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)

from ..types import VIT_NAMES
from .feature_extractor_base import ExtractorBase
from .imagenet_normalization import ImageNetNormalization


class ViTFeatureExtractor(ExtractorBase):
    def __init__(
        self,
        network_name: VIT_NAMES,
        trainable: bool = True,
    ):
        """ViT feature extractor

        Args:
            network_name (VIT_NAMES): torchvision ViT model
            trainable (bool, optional): backbone is trainable or not. Defaults to True.
        """
        super().__init__(network_name, None, trainable)
        self.normalization = ImageNetNormalization()

    @property
    def feature_dims(self) -> list[int]:
        return {
            "b16": [768] * 12,
            "b32": [768] * 12,
            "l16": [1024] * 24,
            "l32": [1024] * 24,
        }[self.network_name]

    def get_weight(self, network_name: str) -> Weights:
        return {
            "b16": ViT_B_16_Weights.DEFAULT,
            "b32": ViT_B_32_Weights.DEFAULT,
            "l16": ViT_L_16_Weights.DEFAULT,
            "l32": ViT_L_32_Weights.DEFAULT,
        }[network_name]

    def build_feature_extractor(self, network_name: str, weights: Weights) -> nn.Module:
        model_class = getattr(
            torchvision.models,
            f"vit_{network_name[0]}_{network_name[1:]}",
        )
        feature_extractor = model_class(weights=weights)

        del feature_extractor.heads

        if not self.trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def forward(
        self,
        img: torch.Tensor,
        target_layer_names: set[int],
    ) -> list[torch.Tensor]:
        features = []

        img = self.normalization(img)

        ### copy from torchvision.models.vision_transformer.VisionTransformer
        # Reshape and permute the input tensor
        y = self.feature_extractor._process_input(img)
        b = y.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.feature_extractor.class_token.expand(b, -1, -1)
        y = torch.cat([batch_class_token, y], dim=1)

        ### copy from torchvision.models.vision_transformer.Encoder
        torch._assert(
            y.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {y.shape}"
        )
        y = y + self.feature_extractor.encoder.pos_embedding
        y = self.feature_extractor.encoder.dropout(y)

        for i, layer in enumerate(self.feature_extractor.encoder.layers):
            y = layer(y)
            if i in target_layer_names:
                # only patch tokens
                features.append(y[:, 1:, :])

        return features
