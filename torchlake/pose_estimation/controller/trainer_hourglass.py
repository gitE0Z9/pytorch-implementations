from typing import Iterable

import torch
from torch import nn
from torchlake.common.controller.trainer import RegressionTrainer
from torchlake.common.utils.numerical import build_gaussian_heatmap


class StackedHourglassTrainer(RegressionTrainer):
    def __init__(self, sigma: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, y = row
        y: torch.Tensor = y.to(self.device)

        coords, masks = y[..., :-1], y[..., -1, None]

        # shift coordinates of y to match grids of indexing 'ij'
        coords = coords.flip(-1)

        # coordinated scaled to predicted heatmap
        scales = torch.Tensor(
            [s_x / s_yhat for s_x, s_yhat in zip(x.shape[2:], y_hat.shape[3:])]
        ).to(self.device)
        coords /= scales[None, None, :]

        # y_hat shape: batch, stack, channel, ...spatial
        # TODO(won't do): multi instance shape: batch, instance, stack, channel
        coords = build_gaussian_heatmap(
            coords,
            spatial_shape=y_hat.shape[3:],
            sigma=self.sigma,
            effective=True,
            kde=True,
        ).exp()
        coords = coords.unsqueeze(1).expand_as(y_hat)

        return criterion(y_hat, coords.float())
