from typing import Sequence
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop


class Decoder(nn.Module):

    def __init__(
        self,
        shallow_input_channel: int,
        deep_input_channel: int,
        output_channel: int,
    ):
        """Decoder in paper [1802.02611v3]

        Args:
            input_channels (Sequence[int]): input channels, one for shallow, another for deep
            output_channel (int): output channel
        """
        super().__init__()
        self.conv = Conv2dNormActivation(shallow_input_channel, deep_input_channel, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.head = nn.Conv2d(2 * deep_input_channel, output_channel, 3, padding=1)

    def forward(self, shallow_x: torch.Tensor, deep_x: torch.Tensor) -> torch.Tensor:
        cropper = CenterCrop(shallow_x.shape[2:])
        y = torch.cat([self.conv(shallow_x), cropper(self.upsample(deep_x))], 1)
        y = self.head(y)
        return y
