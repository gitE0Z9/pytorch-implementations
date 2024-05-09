import torch
from torch import nn


class SqueezeExcitation2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        reduction_ratio: float = 1,
        activations: tuple[nn.Module | None] = (nn.ReLU(True), nn.Sigmoid()),
    ):
        """Squeeze and excitation module from [1709.01507v4]

        Args:
            input_channel (int):input channel size.
            reduction_ratio (float, optional): reduction ratio. Defaults to 1.
            activations (tuple[nn.Module  |  None], optional): activations for fc1 and fc2. Defaults to (nn.ReLU(True), nn.Sigmoid()).
        """
        super(SqueezeExcitation2d, self).__init__()
        self.s = nn.Conv2d(input_channel, input_channel // reduction_ratio, 1)
        self.e = nn.Conv2d(input_channel // reduction_ratio, input_channel, 1)
        self.s_activation, self.e_activation = (
            activation or nn.Identity() for activation in activations
        )

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean((2, 3), keepdim=True)
        y = self.s(y)
        y = self.s_activation(y)
        y = self.e(y)
        return self.e_activation(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.get_attention(x)
