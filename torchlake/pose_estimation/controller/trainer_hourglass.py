from typing import Iterable

import torch
from torch import nn
from torchlake.common.controller.trainer import RegressionTrainer
from torchlake.common.utils.numerical import build_heatmap


class StackedHourglassTrainer(RegressionTrainer):
    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, y = row
        y: torch.Tensor = y.to(self.device)

        scales = torch.Tensor(
            [s_x / s_yhat for s_x, s_yhat in zip(x.shape[2:], y_hat.shape[3:])]
        ).to(self.device)
        y /= scales[None, None, :]
        # y_hat shape: batch, stack, channel, ...spatial
        # TODO(won't do): multi instance shape: batch, instance, stack, channel
        y = build_heatmap(y, spatial_shape=y_hat.shape[3:]).exp()
        y = y.unsqueeze(1).expand_as(y_hat)

        return criterion(y_hat, y.float())
