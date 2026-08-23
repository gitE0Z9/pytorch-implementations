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
        """calculate loss

        Args:
            y_hat (torch.Tensor): shape is (S, B, O)
            row (tuple[Iterable]): shape is (B, C, H, W), (B, S)
            criterion (nn.Module): loss module

        Returns:
            torch.Tensor: loss
        """
        _, y = row

        if isinstance(criterion, nn.CTCLoss):
            pred_len = (
                torch.Tensor([len(y_hat[:, i]) for i in range(y_hat.size(1))])
                .long()
                .to(self.device)
            )
            target_len = (
                torch.Tensor([len(y[i]) for i in range(y.size(0))])
                .long()
                .to(self.device)
            )
            return criterion(y_hat.log_softmax(-1), y, pred_len, target_len)

        return criterion(y_hat, y)
