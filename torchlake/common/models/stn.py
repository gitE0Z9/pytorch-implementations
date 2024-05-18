import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation


class SpatialTransform2d(nn.Module):

    def __init__(self, in_dim: int, latent_dim: int, num_transfomer: int = 1):
        super(SpatialTransform2d, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_dim, latent_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        layers = []
        for _ in range(num_transfomer):
            layer = nn.Sequential(
                Conv2dNormActivation(latent_dim, 32, norm_layer=None),
                nn.Flatten(),
                nn.Linear(32 * 56**2, 6),
            )
            # Initialize the weights/bias with identity transformation
            layer[-1].weight.data.zero_()
            layer[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

            layers.append(layer)

        self.transforms = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        y = self.localization(x)

        transforms = []
        for transform in self.transforms:
            z = transform(y)
            z = F.affine_grid(z.view(-1, 2, 3), x.shape)
            z = F.grid_sample(x, z)
            transforms.append(z)

        return transforms
