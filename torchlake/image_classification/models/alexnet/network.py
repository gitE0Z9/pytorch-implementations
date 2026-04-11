import torch
from torch import nn
import torch.nn.functional as F


class LocalResponseNorm(nn.Module):

    def __init__(
        self,
        input_channel: int,
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

        self.register_buffer(
            "filter",
            torch.ones(input_channel, input_channel, kernel, kernel),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x**2, self.filter, padding=self.kernel // 2)
        return x / (self.k + self.alpha * y) ** self.beta
