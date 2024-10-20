from typing import Iterable

import torch
from torchlake.common.controller.trainer import ClassificationTrainer

from ..models.bilstm_crf import BiLSTMCRF, LinearCRFLoss


class BiLSTMCRFTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: BiLSTMCRF,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, BiLSTMCRF]:
        x, _ = row
        x: torch.Tensor = x.to(self.device)
        output: torch.Tensor = model(x, *args, **kwargs)

        # XXX: hacky
        return output, model

    def _calc_loss(
        self,
        output: tuple[torch.Tensor, BiLSTMCRF],
        row: tuple[Iterable],
        criterion: LinearCRFLoss,
    ):
        _, y = row
        y: torch.Tensor = y.to(self.device)

        output, model = output

        return criterion.forward(output, y, model.head.transition)
