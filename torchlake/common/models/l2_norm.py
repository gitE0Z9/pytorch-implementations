import torch
from torch import nn


class L2Norm(nn.Module):
    def __init__(self, input_channel: int, scale: float = 1.0):
        """L2 normalization in ParseNet [1506.04579v2]
        norm scale of each channel will be learned to cooperate different layer's scale

        Args:
            input_channel (int): input channel size.
            scale (float, optional): init value of output scale. Defaults to 1.0.
        """
        super().__init__()
        self.input_channel = input_channel
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(
            torch.full((1, self.input_channel, 1, 1), self.gamma).float()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=1, p=2, keepdim=True)
        # don't inplace, since grad will be trapped
        x = x / (norm + self.eps)
        return self.weight * x
