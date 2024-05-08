import torch
from torch import nn

from torchvision.ops import Conv2dNormActivation


class Bam2d(nn.Module):

    def __init__(self, in_dim: int, reduction_ratio: float = 16, dilation: int = 4):
        """Bottleneck attention module

        Args:
            in_dim (int): input dimension
            reduction_ratio (float, optional): reduction ratio. Defaults to 1.
        """
        super(Bam2d, self).__init__()
        compressed_dim = in_dim // reduction_ratio

        self.ca = nn.Sequential(
            nn.Conv2d(in_dim, compressed_dim, 1),
            nn.Conv2d(compressed_dim, in_dim, 1),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(in_dim, compressed_dim, 1),
            nn.Conv2d(
                compressed_dim, compressed_dim, 3, dilation=dilation, padding=dilation
            ),
            nn.Conv2d(
                compressed_dim, compressed_dim, 3, dilation=dilation, padding=dilation
            ),
            Conv2dNormActivation(compressed_dim, 1, 1),
        )

    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(x.mean((2, 3), keepdim=True))

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.get_channel_attention(x) + self.get_spatial_attention(x)

        return x * (1 + w.sigmoid())
