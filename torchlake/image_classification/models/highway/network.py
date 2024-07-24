import torch
from torch import nn
from torchlake.common.models import HighwayBlock
from torchvision.ops import Conv2dNormActivation


class HighwayLayer(nn.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
    ):
        super(HighwayLayer, self).__init__()
        self.pre_layer = (
            nn.Identity()
            if in_channel == out_channel
            else Conv2dNormActivation(
                in_channel,
                out_channel,
                1,
            )
        )
        self.block = HighwayBlock(
            nn.Sequential(
                Conv2dNormActivation(
                    out_channel,
                    out_channel,
                    kernel_size,
                )
            ),
            nn.Sequential(
                Conv2dNormActivation(
                    out_channel,
                    out_channel,
                    kernel_size,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_layer(x)
        return self.block(y)
