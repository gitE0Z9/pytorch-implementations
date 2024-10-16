import torch
from torch import nn
from torchlake.common.models import ConvBnRelu


class Block(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        enable_shortcut: bool = True,
        enable_pool: bool = False,
    ):
        super().__init__()
        self.enable_shortcut = enable_shortcut
        self.block = nn.Sequential(
            ConvBnRelu(
                input_channel,
                output_channel,
                kernel,
                padding=kernel // 2,
                dimension="1d",
            ),
            ConvBnRelu(
                output_channel,
                output_channel,
                kernel,
                padding=kernel // 2,
                dimension="1d",
            ),
        )

        if enable_pool:
            self.block.append(nn.MaxPool1d(3, 2, 1))

        if self.enable_shortcut:
            self.downsample = self.build_shortcut(
                input_channel,
                output_channel,
                2 if enable_pool else 1,
            )

    def build_shortcut(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
    ) -> nn.Module:
        """build shortcut

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
        """

        if input_channel == output_channel and stride == 1:
            return nn.Identity()

        return ConvBnRelu(
            input_channel,
            output_channel,
            1,
            stride=stride,
            activation=None,
            dimension="1d",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_shortcut:
            return self.downsample(x) + self.block(x)
        else:
            return self.block(x)
