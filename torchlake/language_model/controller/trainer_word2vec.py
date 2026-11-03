from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import ClassificationTrainer
from ..constants.enum import LossType


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

        # batch*subseq, window-1, embed
        neighbor_size = context.size(1)

        yhat = model(gram, neighbor_size, *args, **kwargs)

        if self.feature_last:
            yhat = yhat.permute(0, -1, *range(1, len(yhat.shape) - 1))

        if model.loss_type == LossType.NEGATIVE_SAMPLING:
            return yhat, model.head
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

        # batch*subseq, 1, embed
        yhat = model(context, *args, **kwargs)

        if self.feature_last:
            yhat = yhat.permute(0, -1, *range(1, len(yhat.shape) - 1))

        if model.loss_type == LossType.NEGATIVE_SAMPLING:
            return yhat, model.head
        if model.loss_type == LossType.HIERARCHICAL_SOFTMAX:
            return yhat, model.head.node_embedding

        return yhat

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ):
        gram, _ = row
        gram: torch.Tensor = gram.to(self.device)

        if isinstance(y_hat, tuple):
            y_hat, embed = y_hat
            return criterion(y_hat, gram, output_embedding=embed)
        else:
            return criterion(y_hat, gram)
