import torch
from torch import nn
from torchlake.common.models import SpatialTransform2d
from torchvision.ops import Conv2dNormActivation


class SpatialTransformNetwork(nn.Module):

    def __init__(
        self,
        in_dim: int = 3,
        latent_dim: int = 32,
        output_size: int = 1,
        dropout_prob: float = 0.5,
    ):
        super(SpatialTransformNetwork, self).__init__()
        self.layers = nn.Sequential(
            SpatialTransform2d(in_dim, latent_dim),
            Conv2dNormActivation(latent_dim, latent_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
            SpatialTransform2d(latent_dim, latent_dim),
            Conv2dNormActivation(latent_dim, latent_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
            SpatialTransform2d(latent_dim, latent_dim),
            Conv2dNormActivation(
                latent_dim,
                latent_dim,
                3,
                padding=1,
                activation_layer=None,
            ),
        )
        self.pool = nn.Flatten()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.pool(y)
        y = self.dropout(y)
        return self.fc(y)
