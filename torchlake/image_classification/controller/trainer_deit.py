from typing import Iterable

import torch
from torch import nn
from torchlake.common.controller.trainer import ClassificationTrainer


class DeiTTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        t_model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = row
        x: torch.Tensor = x.to(self.device)

        return (*model(x), t_model(x))

    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion(*y_hat, y.long())
