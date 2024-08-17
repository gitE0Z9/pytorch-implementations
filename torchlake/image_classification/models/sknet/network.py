import torch
from torch import nn
from torchlake.common.network import ConvBnRelu

from ..resnext.network import BottleNeck as XBottleNeck


class SelectiveKernel2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        reduction_ratio: int = 16,
    ):
        super(SelectiveKernel2d, self).__init__()
        hidden_dim = max(32, output_channel // reduction_ratio)
        self.conv1 = ConvBnRelu(
            input_channel,
            output_channel,
            3,
            padding=1,
            group=32,
        )
        self.conv2 = ConvBnRelu(
            input_channel,
            output_channel,
            3,
            padding=2,
            dilation=2,
            group=32,
        )

        self.s = nn.Sequential(
            nn.Linear(output_channel, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )

        self.attention = nn.Linear(hidden_dim, output_channel * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        # B, C, H, W
        batch_size, c, h, w = y1.shape
        # B, 2, C, H, W
        _y = torch.cat([y1, y2], 1).view(batch_size, 2, c, h, w)

        # B, C, H, W
        y = _y.sum(1)
        # B, C
        y = y.mean((2, 3))
        # B, hidden
        y = self.s(y)

        # B, 2C
        attention = self.attention(y)
        # B, 2, C
        attention = attention.view(batch_size, 2, c).softmax(1)

        # B, 2, C, 1, 1 x B, 2, C, H, W => B, C, H, W
        return attention[:, :, :, None, None].mul(_y).sum(1)


class BottleNeck(XBottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """bottleneck block in sknet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 2 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): activation before block
        """
        super(BottleNeck, self).__init__(
            input_channel,
            block_base_channel,
            stride,
            pre_activation,
        )
        self.block[1] = SelectiveKernel2d(block_base_channel, block_base_channel)
