import torch
from torch import nn


class GanGenerator(nn.Module):
    def __init__(self, latent_dim=128, image_size: int = 28):
        super(GanGenerator, self).__init__()
        self.image_size = image_size
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, self.image_size**2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.generator(x)
        y = y.view(-1, 1, self.image_size, self.image_size)
        return y


class GanDiscriminator(nn.Module):
    def __init__(self, image_size: int):
        super(GanDiscriminator, self).__init__()
        self.image_size = image_size

        self.discriminator = nn.Sequential(
            nn.Linear(image_size**2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.image_size**2)
        y = self.discriminator(y)
        return y
