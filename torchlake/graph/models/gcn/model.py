import torch
import torch.nn.functional as F
from torch import nn

from .network import GCNLayer, GCNResBlock


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(in_dim, hidden_dim),
                GCNLayer(hidden_dim, out_dim),
            ]
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.layers[0](x, a)
        for layer in self.layers[1:]:
            y = F.relu(y, True)
            y = layer(y, a)

        return y


class GCNResidual(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(in_dim, hidden_dim),
                GCNResBlock(hidden_dim, hidden_dim),
                GCNResBlock(hidden_dim, hidden_dim),
                GCNResBlock(hidden_dim, hidden_dim),
                GCNLayer(hidden_dim, out_dim),
            ]
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.layers[0](x, a)
        for layer in self.layers[1:]:
            y = F.relu(y, True)
            y = layer(y, a)

        return y
