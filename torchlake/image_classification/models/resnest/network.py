import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from ..resnet.network import BottleNeck as RBottleNeck


class SplitAttention2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        reduction_ratio: int = 4,
        cardinality: int = 1,
        radix: int = 2,
    ):
        super(SplitAttention2d, self).__init__()
        self.radix = radix
        hidden_dim = max(32, input_channel * radix // reduction_ratio)
        self.conv = Conv2dNormActivation(
            input_channel,
            output_channel * radix,
            groups=cardinality * radix,
        )

        self.s = Conv2dNormActivation(output_channel, hidden_dim, 1, groups=cardinality)

        self.attention = Conv2dNormActivation(
            hidden_dim,
            output_channel * radix,
            1,
            groups=cardinality,
            norm_layer=None,
            activation_layer=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, r * C, H, W
        y = self.conv(x)

        # B, r*C, H, W
        batch_size, total_channels, h, w = y.shape
        c = total_channels // self.radix
        # B, r, C, H, W
        _y = y.view(batch_size, self.radix, c, h, w)

        # B, C, H, W
        y = _y.sum(1)
        # B, C, 1, 1
        y = y.mean((2, 3), keepdim=True)
        # B, hidden, 1, 1
        y = self.s(y)

        # B, r * C, 1, 1
        attention = self.attention(y)
        # B, r, C, 1, 1
        attention = attention.view(
            batch_size,
            self.radix,
            c,
            1,
            1,
        ).softmax(1)

        # B, r, C, 1, 1 x B, r, C, H, W => B, C, H, W
        return attention.mul(_y).sum(1)


class BottleNeck(RBottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        pre_activation: bool = False,
    ):
        """bottleneck block in resnest
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 2 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            pre_activation (bool, Defaults False): activation before block
        """
        super(BottleNeck, self).__init__(
            input_channel,
            block_base_channel,
            pre_activation,
        )
        self.block[1] = SplitAttention2d(block_base_channel, block_base_channel)
