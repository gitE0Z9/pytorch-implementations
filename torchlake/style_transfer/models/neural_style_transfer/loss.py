from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models.feature_extractor_base import ExtractorBase


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d = x.shape
    y = x.reshape(a, b, c * d)
    y = torch.bmm(y, y.transpose(1, 2))
    y = y / (b * c * d)

    return y


class NeuralStyleTransferLoss(nn.Module):
    def __init__(
        self,
        backbone: ExtractorBase,
        content_layer_name: str,
        style_layer_names: Sequence[str],
        content_weight: float,
        style_weight: float,
        norm_generated: bool = True,
        return_all_loss: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.content_layer_name = content_layer_name
        self.style_layer_names = style_layer_names
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.norm_generated = norm_generated
        self.return_all_loss = return_all_loss

    def set_style_features(self, style: torch.Tensor):
        with torch.inference_mode():
            self.style_features = self.backbone(style, self.style_layer_names)

    def calc_style_loss(
        self,
        feature: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(gram_matrix(feature), gram_matrix(style).detach())

    def calc_content_loss(
        self,
        feature: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(feature, content.detach())

    def forward(
        self,
        yhat: torch.Tensor,
        content: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        features = self.backbone(
            yhat,
            self.style_layer_names,
            normalization=self.norm_generated,
        )

        content_loss = 0
        if content is not None:
            if self.content_layer_name in self.style_layer_names:
                feature_for_content = features[
                    self.style_layer_names.index(self.content_layer_name)
                ]
            else:
                feature_for_content = self.backbone(
                    yhat, [self.content_layer_name]
                ).pop()

            content_feature = self.backbone(
                # [:, :3] for texture net
                content[:, :3],
                [self.content_layer_name],
            ).pop()
            content_loss = self.calc_content_loss(feature_for_content, content_feature)

        style_loss = 0
        for feature, style_feature in zip(features, self.style_features):
            style_loss += self.calc_style_loss(feature, style_feature)

        total_loss = self.content_weight * content_loss + self.style_weight * style_loss

        if self.return_all_loss:
            return total_loss, content_loss, style_loss
        else:
            return total_loss
