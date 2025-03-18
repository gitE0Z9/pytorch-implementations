from itertools import pairwise
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class DarkNetBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        base_channel: int,
        expansion_ratios: list[int] = [1, 2],
    ):
        """
        DarkNet53 block

        Args:
            input_channel (int): input channel size
            base_channel (int): the smallest channel
            expansion_ratios (list[int]): expansion ratio of channels of each layer
        """
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                Conv2dNormActivation(
                    base_channel * in_r if i != 0 else input_channel,
                    base_channel * out_r,
                    kernel_size=1 if out_r == 1 else 3,
                    activation_layer=lambda: nn.LeakyReLU(0.1),
                    inplace=None,
                )
                for i, (in_r, out_r) in enumerate(pairwise([0, *expansion_ratios]))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
