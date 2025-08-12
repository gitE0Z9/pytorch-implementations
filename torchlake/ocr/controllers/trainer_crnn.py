from typing import Iterable
import torch
from torch import nn
from torchlake.common.controller.trainer import ClassificationTrainer


class CRNNTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row

        pred_len = torch.Tensor([len(y_hat)]).long().to(self.device)
        target_len = torch.Tensor([len(y)]).long().to(self.device)
        return criterion(y_hat, y, pred_len, target_len)
