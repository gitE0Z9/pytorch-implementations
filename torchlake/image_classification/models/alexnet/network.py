import torch
from torch import nn


class LocalResponseNorm(nn.Module):

    def __init__(
        self,
        k: int = 2,
        kernel: int = 5,
        alpha: float = 1e-4,
        beta: float = 0.75,
    ):
        super().__init__()
        self.k = k
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta

        self.layer = nn.Unfold(kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.layer(x)
        y = y.view(b, c, -1, h, w)
        return x / (self.k + self.alpha * y.square().sum(2)) ** self.beta
