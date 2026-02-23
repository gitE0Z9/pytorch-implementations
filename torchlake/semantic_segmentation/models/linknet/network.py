import torch
from torch import nn

from torchlake.common.models import ConvBNReLU
from torchlake.common.models.residual import ResBlock
from torchlake.image_classification.models.resnet.network import ConvBlock


class EncoderBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        """Two bottlenecks of ResNet18

        Args:
            input_channel (int, optional): input channel size.
            output_channel (int): output channel size.
        """
        super().__init__()
        self.layers = nn.Sequential(
            ResBlock(
                input_channel,
                output_channel,
                block=ConvBlock(input_channel, output_channel, stride=2),
                stride=2,
            ),
            ResBlock(
                output_channel,
                output_channel,
                block=ConvBlock(output_channel, output_channel),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        """The bottleneck of ResNet but the middle layer is deconvolution

        Args:
            input_channel (int, optional): input channel size.
            output_channel (int): output channel size.
        """
        super().__init__()
        h = input_channel // 4

        self.layers = nn.Sequential(
            ConvBNReLU(input_channel, h, 1),
            ConvBNReLU(h, h, 4, stride=2, padding=1, deconvolution=True),
            ConvBNReLU(h, output_channel, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
