from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import ClassificationTrainer


class InceptionTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: list[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        loss = 0
        for ele in y_hat:
            loss += criterion(ele, y.long())
        return loss
