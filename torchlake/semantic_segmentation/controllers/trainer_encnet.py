from typing import Iterable
import torch
from torch import nn
from ..models.encnet.loss import EncNetLoss
from torchlake.common.controller.trainer import ClassificationTrainer


class EncNetTrainer(ClassificationTrainer):

    def _calc_loss(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        row: tuple[Iterable],
        criterion: EncNetLoss,
    ):
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion.forward(*output, y.long())
