from typing import Iterable

import torch
from torch import nn
from torchlake.common.controller.trainer import RegressionTrainer
from torchlake.common.utils.numerical import build_gaussian_heatmap


class StackedHourglassTrainer(RegressionTrainer):
    def __init__(
        self,
        sigma: float = 1,
        effective_range: int = 3,
        amplitude: float = 1,
        visible_only: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.effective_range = effective_range
        self.amplitude = amplitude
        self.visible_only = visible_only

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, y = row
        y: torch.Tensor = y.to(self.device)

        # B, C, 2
        coords = y[:, :, :-1]

        # shift coordinates of y to match grids of indexing 'ij'
        coords = coords.flip(-1)

        # coordinated scaled to predicted heatmap
        scales = torch.Tensor(
            [s_x / s_yhat for s_x, s_yhat in zip(x.shape[2:], y_hat.shape[3:])]
        ).to(self.device)
        coords /= scales[None, None, :]

        # y_hat shape: batch, stack, channel, ...spatial
        # not support multi instance, shape: batch, instance, stack, channel, ...spatial
        heatmaps = build_gaussian_heatmap(
            coords,
            spatial_shape=y_hat.shape[3:],
            sigma=self.sigma,
            effective_range=self.effective_range,
            normalized=False,
            amplitude=self.amplitude,
        ).exp()
        # B, C, H, W => B, 1, C, H, W => B, A, C, H, W
        heatmaps = heatmaps.unsqueeze(1).expand_as(y_hat)

        if self.visible_only:
            # B, 1, C, 1, 1
            masks = y[:, None, :, -1, None, None]
            # remove invisible from loss
            visible = 1 - masks
            return criterion(y_hat * visible, heatmaps.float() * visible)
        else:
            return criterion(y_hat, heatmaps.float())
