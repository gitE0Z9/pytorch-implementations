import torch
from torch import nn

from torchlake.common.controller.trainer import TrainerBase


class FastStyleTransferTrainer(TrainerBase):
    def _predict(
        self,
        row: torch.Tensor,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        x = row
        x = x.to(self.device)

        return model(x)

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x = row
        x = x.to(self.device)

        return criterion(y_hat, x)
