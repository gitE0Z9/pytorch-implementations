import torch
from torch import nn
from torchlake.common.models import ConvBNReLU

from ..resnet.network import BottleNeck as BottleNeckBase


class SelectiveKernel2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        reduction_ratio: int = 16,
        pre_activation: bool = False,
    ):
        """selective kernel convolution in paper [1903.06586v2]
        split: 3x3 conv and dilated 3x3 conv (as 5x5)
        fuse: global average pooling of summed branches -> fc
        select: softmax attention over branches

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            reduction_ratio (int, optional): reduction ratio r of fuse fc, hidden dimension is max(32, output_channel // r). Defaults to 16.
            pre_activation (bool, optional): put activation before convolution layer of branches in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__()
        hidden_dim = max(32, output_channel // reduction_ratio)
        self.conv1 = ConvBNReLU(
            input_channel,
            output_channel,
            3,
            padding=1,
            group=32,
            conv_last=pre_activation,
        )
        self.conv2 = ConvBNReLU(
            input_channel,
            output_channel,
            3,
            padding=2,
            dilation=2,
            group=32,
            conv_last=pre_activation,
        )

        self.fc = nn.Sequential(
            nn.Linear(output_channel, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )

        self.attention = nn.Linear(hidden_dim, output_channel * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        b, c, _, _ = y1.shape
        # B, 2, C, H, W
        y = torch.stack((y1, y2), 1)

        # B, C
        z = y.sum(1).mean((2, 3))
        # B, hidden
        z = self.fc(z)
        # B, 2C
        a = self.attention(z)
        # B, 2, C
        a = a.view(b, 2, c).softmax(1)

        # B, 2, C x B, 2, C, H, W => B, C, H, W
        return torch.einsum("bkc,bkchw->bchw", a, y)


class BottleNeck(BottleNeckBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 2,
        pre_activation: bool = False,
    ):
        """bottleneck block in sknet
        kernel: 1x1 -> selective kernel -> 1x1
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): expansion factor of base channel. Defaults to 2.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        self.layers[1] = SelectiveKernel2d(
            hidden_dim,
            hidden_dim,
            pre_activation=pre_activation,
        )
