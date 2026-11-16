import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.mixins.network import SEMixin

from ..resnet.network import BottleNeck as BottleNeckBase


class CBAM2d(nn.Module):

    def __init__(self, in_dim: int, reduction_ratio: float = 1):
        """Convolution block attention module

        Args:
            in_dim (int): input dimension
            reduction_ratio (float, optional): reduction ratio. Defaults to 1.
        """
        super().__init__()
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


class BottleNeck(SEMixin, BottleNeckBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        pre_activation: bool = False,
        reduction_ratio: int = 16,
    ):
        """bottleneck block in cbam-resnet
        kernel: 1x1 -> 3x3 -> 1x1 -> cbam
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 4.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            reduction_ratio (int, optional): reduction ratio of cbam. Defaults to 16.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        self.se = CBAM2d(hidden_dim * expansion_factor, reduction_ratio)
