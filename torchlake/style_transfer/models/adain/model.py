import torch
from torch import nn
from torchlake.common.models import VGGFeatureExtractor

from .network import AdaIn2d, AdaInDecoder


class AdaInTrainer(nn.Module):
    def __init__(
        self,
        feature_extractor: VGGFeatureExtractor,
        style_layer_names: list[str],
    ):
        super(AdaInTrainer, self).__init__()
        self.style_layer_names = style_layer_names
        self.encoder = feature_extractor
        self.decoder = AdaInDecoder()

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float = 1,
    ) -> torch.Tensor:
        last_layer = [self.style_layer_names[-1]]
        content_feature = self.encoder(content, last_layer)[0]
        style_feature = self.encoder(style, last_layer)[0]

        adain = AdaIn2d(content_feature, style_feature)
        normalized = (1 - alpha) * content_feature + alpha * adain(content_feature)

        generated = self.decoder(normalized)

        return generated

    def style_interpolation_forward(
        self,
        content: torch.Tensor,
        styles: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        last_layer = [self.style_layer_names[-1]]
        content_feature = self.encoder(content, last_layer)[0]
        styles_features = self.encoder(styles, last_layer)[0]

        normalized = torch.zeros_like(content_feature)
        for style_feature, weight in zip(styles_features, weights):
            adain = AdaIn2d(content_feature, style_feature.unsqueeze(0))
            normalized += weight * adain(content_feature)

        generated = self.decoder(normalized)

        return generated

    def loss_forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        content_feature = self.encoder(content, [self.style_layer_names[-1]])[0]
        style_features = self.encoder(style, self.style_layer_names)

        adain = AdaIn2d(content_feature, style_features[-1])
        normalized = adain(content_feature)

        generated = self.decoder(normalized)

        generated_features = self.encoder(generated, self.style_layer_names)

        return generated_features, normalized, style_features
