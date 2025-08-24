from typing import Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import TrainerBase


class YOLOV2Trainer(TrainerBase):
    seen = 0

    def reset_seen(self):
        self.seen = 0

    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, y = row

        self.seen += len(x)

        return criterion(y_hat, y, seen=self.seen)
