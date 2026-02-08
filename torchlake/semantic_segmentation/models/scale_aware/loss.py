from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ScaleAwareLoss(nn.Module):
    def forward(self, outputs: Sequence[torch.Tensor], gt: torch.Tensor):
        loss = 0
        for output in outputs:
            loss += F.cross_entropy(output, gt)

        return loss
