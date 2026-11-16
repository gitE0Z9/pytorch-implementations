import torch
from torch import nn


class HighwayLayer(nn.Module):

    def __init__(self, block: nn.Module, gate: nn.Module):
        super().__init__()
        self.block = block
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_y = self.gate(x).sigmoid()
        return (1 - gate_y) * x + gate_y * self.block(x)
