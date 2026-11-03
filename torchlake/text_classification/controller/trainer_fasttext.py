from typing import Any, Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import ClassificationTrainer
from torchlake.language_model.constants.enum import LossType
from ..models.fasttext import FastText


class FastTextTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: FastText,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        (ngrams, words, word_spans), _ = row
        x = (
            [it.to(self.device) for it in ngrams],
            words.to(self.device),
            [it.to(self.device) for it in word_spans],
        )

        yhat = model(*x, *args, **kwargs)

        if model.loss_type == LossType.HIERARCHICAL_SOFTMAX:
            return yhat, model.head.node_embedding

        return yhat

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ):
        _, context = row
        context: torch.Tensor = context.to(self.device)

        if isinstance(y_hat, tuple):
            y_hat, embed = y_hat
            return criterion(y_hat, context, output_embedding=embed)
        else:
            return criterion(y_hat, context)
