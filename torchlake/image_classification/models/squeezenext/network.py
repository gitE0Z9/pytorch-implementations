import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class BottleNeck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
    ):
        """Bottleneck in paper [1803.10615v2]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
        """
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                output_channel // 2,
                1,
                stride=stride,
            ),
            Conv2dNormActivation(
                output_channel // 2,
                output_channel // 4,
                1,
            ),
            Conv2dNormActivation(
                output_channel // 4,
                output_channel // 2,
                (3, 1),
                padding=(1, 0),
            ),
            Conv2dNormActivation(
                output_channel // 2,
                output_channel // 2,
                (1, 3),
                padding=(0, 1),
            ),
            Conv2dNormActivation(
                output_channel // 2,
                output_channel,
                1,
                activation_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
