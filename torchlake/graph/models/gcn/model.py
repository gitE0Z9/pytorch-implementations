import torch
from torch import nn
from .network import GcnLayer, GcnResBlock
import torch.nn.functional as F


class Gcn(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(Gcn, self).__init__()

        self.layer1 = GcnLayer(in_dim, hidden_dim)
        self.layer2 = GcnLayer(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.layer1(x, a)
        y = F.relu(y, True)
        y = self.layer2(y, a)

        return y


class GcnResidualVersion(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(GcnResidualVersion, self).__init__()

        self.layer1 = GcnLayer(in_dim, hidden_dim)
        self.layer2 = GcnResBlock(hidden_dim, hidden_dim)
        self.layer3 = GcnResBlock(hidden_dim, hidden_dim)
        self.layer4 = GcnResBlock(hidden_dim, hidden_dim)
        self.layer5 = GcnLayer(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.layer1(x, a)
        y = F.relu(y, True)
        y = self.layer2(y, a)
        y = F.relu(y, True)
        y = self.layer3(y, a)
        y = F.relu(y, True)
        y = self.layer4(y, a)
        y = F.relu(y, True)
        y = self.layer5(y, a)

        return y
