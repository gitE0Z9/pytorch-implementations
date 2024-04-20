from typing import Iterable

from torch import nn
from torchlake.common.controller.trainer import TrainerBase


class SkipGramTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        criterion: nn.Module,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        # batch, context-1, subseq, embed
        output = model(gram, *args, **kwargs)

        loss = criterion(output, context)

        return loss


class CbowTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        criterion: nn.Module,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)
        context = context.to(self.device)

        # batch, 1, subseq, embed
        output = model(context, *args, **kwargs)

        loss = criterion(output, gram)

        return loss
