from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import ClassificationTrainer

from ..mixins.curriculum import CurriculumMixin
from ..models.base.rnn_generator import RNNGenerator


class RNNGeneratorTrainer(CurriculumMixin, ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: RNNGenerator,
        *args,
        **kwargs,
    ):
        x = row
        x = x.to(self.device)

        if "teacher_forcing_ratio" in kwargs:
            teacher_forcing_ratio = kwargs.pop("teacher_forcing_ratio")
        else:
            teacher_forcing_ratio = self.teacher_forcing_raio

        output = model.forward(
            x,
            teacher_forcing_ratio=teacher_forcing_ratio,
            *args,
            **kwargs,
        )

        if self.feature_last:
            output = output.permute(0, -1, *range(1, len(output.shape) - 1))

        return output

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x = row
        x: torch.Tensor = x.to(self.device)

        # early stopping since pad are ignored when training
        return criterion(y_hat, x.long()[:, : y_hat.size(2)])
