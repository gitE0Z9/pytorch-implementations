import torch
from torch import nn
import torch.nn.functional as F


class PrototypicalNetworkLoss(nn.Module):
    def __init__(self):
        super(PrototypicalNetworkLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(-x, y) + x.mean()
