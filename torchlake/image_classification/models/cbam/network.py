import torch
from torch import nn
from torchlake.common.mixins.network import SeMixin
from torchvision.ops import Conv2dNormActivation

from ..resnet.network import BottleNeck


class Cbam2d(nn.Module):

    def __init__(self, in_dim: int, reduction_ratio: float = 1):
        """Convolution block attention module

        Args:
            in_dim (int): input dimension
            reduction_ratio (float, optional): reduction ratio. Defaults to 1.
        """
        super(Cbam2d, self).__init__()
        self.ca = nn.Sequential(
            Conv2dNormActivation(
                in_dim,
                in_dim // reduction_ratio,
                1,
                norm_layer=None,
            ),
            Conv2dNormActivation(
                in_dim // reduction_ratio,
                in_dim,
                1,
                norm_layer=None,
                activation_layer=None,
            ),
        )
        self.sa = Conv2dNormActivation(
            2,
            1,
            7,
            norm_layer=None,
            activation_layer=None,
        )

    def get_poolings(
        self,
        x: torch.Tensor,
        dims: int | tuple[int],
    ) -> tuple[torch.Tensor]:
        avg_pooled = x.mean(dims, keepdim=True)
        max_pooled = x.amax(dims, keepdim=True)

        return avg_pooled, max_pooled

    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel_size, _, _ = x.shape
        return (
            self.ca(torch.cat(self.get_poolings(x, (2, 3)), 0))
            .view(2, batch_size, channel_size, 1, 1)
            .sum(0)
            .sigmoid()
        )

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(torch.cat(self.get_poolings(x, 1), 1)).sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.get_channel_attention(x)
        y = y * self.get_spatial_attention(y)

        return y


class BottleNeck(SeMixin, BottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
    ):
        """bottleneck block in se-resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, Defaults False): put activation before transformation [1603.05027v3]
        """
        super(BottleNeck, self).__init__(
            input_channel,
            block_base_channel,
            stride,
            pre_activation,
        )
        self.se = Cbam2d(block_base_channel * 4, 16)
