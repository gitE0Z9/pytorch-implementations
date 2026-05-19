from typing import Sequence

import torch
from torch import nn

from torchlake.common.controller.trainer import TrainerBase


class TextureTrainer(TrainerBase):
    def set_texture_synthesis(self, texture_synthesis: bool):
        self.texture_synthesis = texture_synthesis

    def _predict(self, row: Sequence[torch.Tensor], model: nn.Module, *args, **kwargs):
        x = row
        x = [it.to(self.device) for it in x]

        return model(x)

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: Sequence[torch.Tensor],
        criterion: nn.Module,
    ) -> torch.Tensor:
        if getattr(self, "texture_synthesis", False):
            x = None
        else:
            x = row
            x = x[0].to(self.device)

        return criterion(y_hat, x)
