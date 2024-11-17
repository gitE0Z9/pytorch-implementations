import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class BottleNeck(nn.Module):

    def __init__(self, channel: int, num_blocks: int):
        """DarkNet19 BottleNeck
        K: 3 -> 1 -> 3
        C: c -> c/2 -> c

        Args:
            channel (int): channel
            num_blocks (int): num blocks
        """
        super().__init__()
        blocks = []
        for block_idx in range(num_blocks):
            channels = [channel // 2, channel]
            kernel = 3

            if block_idx % 2 != 0:
                channels.reverse()
                kernel = 1

            in_channels, out_channels = channels
            blocks.append(
                Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel,
                    activation_layer=lambda: nn.LeakyReLU(0.1),
                    inplace=None,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
