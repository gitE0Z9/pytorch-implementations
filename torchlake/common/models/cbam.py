import torch
from torch import nn

from torchvision.ops import Conv2dNormActivation


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
