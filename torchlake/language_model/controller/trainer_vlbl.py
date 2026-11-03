from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import TrainerBase

from ..models.vlbl.loss import NCE
from ..models.vlbl.model import IVLBL, VLBL


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
        # batch, 1, subseq
        return model(context, gram), model

    def _calc_loss(
        self,
        output: tuple[torch.Tensor, nn.Module],
        row: tuple[Iterable],
        criterion: NCE,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        y_hat, model = output
        return criterion(y_hat, context, gram, model=model)


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
        # batch, neighbor_size, subseq
        return model(gram, context), model

    def _calc_loss(
        self,
        output: tuple[torch.Tensor, nn.Module],
        row: tuple[Iterable],
        criterion: NCE,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        y_hat, model = output
        return criterion(y_hat, gram, context, model=model)
