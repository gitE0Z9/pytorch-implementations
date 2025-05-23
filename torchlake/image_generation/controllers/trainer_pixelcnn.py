from typing import Iterable

import torch
from torch import nn
from torchlake.common.controller.trainer import ClassificationTrainer


class PixelCNNTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        # B, C, 256, H, W => B*C, 256, H, W
        return criterion(y_hat.view(-1, *y_hat.shape[2:]), y.long())
