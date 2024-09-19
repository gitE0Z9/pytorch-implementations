from typing import Iterable

import torch
from torchlake.common.controller.trainer import TrainerBase

from ..models.glove.loss import GloVeLoss
from ..models.glove.model import GloVe


class GloVeTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: GloVe,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        # batch*subseq, neighbor_size
        return model.forward(gram, context)

    def _calc_loss(
        self,
        output: torch.Tensor,
        row: tuple[Iterable],
        criterion: GloVeLoss,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        return criterion.forward(gram, context, output)
