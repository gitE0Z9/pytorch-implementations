from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import TrainerBase


class PrototypicalTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_set, support_set = row
        query_set, support_set = query_set.to(self.device), support_set.to(self.device)

        return model(query_set, support_set)

    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        return criterion(y_hat)
