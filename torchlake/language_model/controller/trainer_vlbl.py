from typing import Iterable

import torch
from torchlake.common.controller.trainer import TrainerBase

from ..models.vlbl.loss import NCE
from ..models.vlbl.model import IVLBL, VLBL


class IVLBLTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: IVLBL,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        # XXX: hacky
        # batch, 1, subseq
        return model.forward(gram, context), model

    def _calc_loss(
        self,
        output: torch.Tensor,
        row: tuple[Iterable],
        criterion: NCE,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        y_hat, model = output
        return criterion.forward(model, gram, context, y_hat)


class VLBLTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: VLBL,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        # XXX: hacky
        # batch, 1, subseq, embed
        return model.forward(context, gram), model

    def _calc_loss(
        self,
        output: torch.Tensor,
        row: tuple[Iterable],
        criterion: NCE,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        y_hat, model = output
        return criterion.forward(model, context, gram, y_hat)
