import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class FireModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        squeeze_ratio: float = 1,
        expand_ratio: float = 1,
    ):
        """Bottleneck in paper [1707.01083v2]

        Args:
            input_channel (int): input channel size
            squeeze_ratio (int, optional): ratio to compress channel. Defaults to 1.
            expand_ratio (int, optional): ratio to expand channel. Defaults to 1.
        """
        super(FireModule, self).__init__()
        compressed_channel = int(squeeze_ratio * input_channel)
        expanded_channel = int(expand_ratio * compressed_channel)

        self.squeeze = Conv2dNormActivation(
            input_channel,
            compressed_channel,
            1,
            norm_layer=None,
        )
        self.expand1 = Conv2dNormActivation(
            compressed_channel,
            expanded_channel,
            1,
            norm_layer=None,
        )
        self.expand2 = Conv2dNormActivation(
            compressed_channel,
            expanded_channel,
            3,
            norm_layer=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        return torch.cat([self.expand1(y), self.expand2(y)], 1)
