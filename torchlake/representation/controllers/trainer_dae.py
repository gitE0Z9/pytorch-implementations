from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import TrainerBase


class DenoisingAutoEncoderTrainer(TrainerBase):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        x, _, z = row
        x, z = x.to(self.device), z.to(self.device)

        return model(x + z, *args, **kwargs)

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x = row[0]
        x = x.to(self.device)

        return criterion(y_hat, x.float())
