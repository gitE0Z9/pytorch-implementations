import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.utils.numerical import build_heatmap


class StackedHourglassLoss(nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        num_stack = pred.size(1)

        gt = build_heatmap(gt, spatial_shape=pred.shape[3:])
        return F.mse_loss(
            pred,
            gt.unsqueeze(1).repeat(1, num_stack, *(1,) * (gt.ndim - 1)),
        )
