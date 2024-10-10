import torch
from torch import nn


class L2Norm(nn.Module):
    def __init__(self, input_channel: int, scale: float = 1.0):
        super(L2Norm, self).__init__()
        self.input_channel = input_channel
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(
            torch.full((1, self.input_channel, 1, 1), self.gamma)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=1, p=2, keepdim=True)
        # don't inplace, since grad will be trapped
        x = x / (norm + self.eps)
        return self.weight * x
