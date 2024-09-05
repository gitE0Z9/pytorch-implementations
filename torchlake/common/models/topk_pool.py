import torch
from torch import nn


class KmaxPool1d(nn.Module):

    def __init__(self, topk: int):
        super(KmaxPool1d, self).__init__()
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.topk(self.topk, -1)[0]
