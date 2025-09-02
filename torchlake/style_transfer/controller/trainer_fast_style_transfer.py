from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import DoNothingTrainer
from torchlake.common.models.feature_extractor_base import ExtractorBase

from ..models.fast_style_transfer.model import FastStyleTransfer


class FastStyleTransferTrainer(DoNothingTrainer):
    def _predict(
        self,
        row,
        model: FastStyleTransfer,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        content = row
        content = content.to(self.device)

        feature_extractor: ExtractorBase = kwargs.pop("feature_extractor")
        content_layer_names: list[str] = kwargs.pop("content_layer_names")
        assert len(content_layer_names) == 1, "only support one content layer"
        style_layer_names: list[str] = kwargs.pop("style_layer_names")

        with torch.no_grad():
            # TODO: precomputed
            content_feature = feature_extractor(
                content,
                content_layer_names,
                normalization=True,
            ).pop()

        generated: torch.Tensor = model(content)
        generated_features: list[torch.Tensor] = feature_extractor(
            generated,
            style_layer_names,
            normalization=False,
        )

        return content_feature, kwargs.pop("style_features"), generated_features

    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        _: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        content_feature, style_features, generated_features = y_hat
        return criterion(content_feature, style_features, generated_features)
