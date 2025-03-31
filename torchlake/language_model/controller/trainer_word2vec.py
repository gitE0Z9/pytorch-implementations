from typing import Iterable

from torch import nn
import torch
from torchlake.common.controller.trainer import ClassificationTrainer


class SkipGramTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)

        # batch, context-1, subseq, embed
        neighbor_size = context.size(1)

        return model(gram, neighbor_size, *args, **kwargs)


class CBOWTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        _, context = row
        context = context.to(self.device)

        # batch, 1, subseq, embed
        return model(context, *args, **kwargs)

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ):
        gram, _ = row
        gram: torch.Tensor = gram.to(self.device)

        return criterion(y_hat, gram)
