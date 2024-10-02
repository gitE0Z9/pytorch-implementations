import torch
from torch import nn


class UpSamplingLegacy(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
    ):
        """Upsampling block of legacy FCN

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            kernel (int): kernel size of deconvolution layer
            stride (int, optional): stride of deconvolution layer. Defaults to 1.
        """
        super(UpSamplingLegacy, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel, stride=stride),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSampling(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int = 1,
    ):
        """Upsampling block of FCN

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            kernel (int): kernel size of deconvolution layer
            stride (int, optional): stride of deconvolution layer. Defaults to 1.
        """
        super(UpSampling, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                input_channel,
                output_channel,
                kernel,
                stride=stride,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
