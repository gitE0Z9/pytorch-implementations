import torch
from torch import nn


class TopKMaxPool1d(nn.Module):

    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.topk(self.topk, -1)[0]
