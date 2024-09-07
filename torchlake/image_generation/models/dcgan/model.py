import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class DcganGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128, init_scale: int = 7):
        super(DcganGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.init_scale = init_scale

        self.noise_projection = nn.Linear(latent_dim, latent_dim * init_scale**2)

        self.generator = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dNormActivation(latent_dim, 128, 3),
            nn.Upsample(scale_factor=2),
            Conv2dNormActivation(128, 64, 3),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.noise_projection(x)
        y = y.view(-1, self.latent_dim, self.init_scale, self.init_scale)
        y = self.generator(y)

        return y


class DcganDiscriminator(nn.Module):
    def __init__(self):
        super(DcganDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            ConvBnRelu(3, 32, 3, padding=1),
            ConvBnRelu(32, 64, 3, padding=1),
            ConvBnRelu(64, 128, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.clf = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.discriminator(x)
        y = torch.flatten(y, start_dim=1)
        y = self.clf(y)
        return y
