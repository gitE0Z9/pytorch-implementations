import torch
from torch import nn


class Vae(nn.Module):
    def __init__(
        self,
        image_size: int = 28 * 28,
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ):
        super(Vae, self).__init__()
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.sigma_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_size),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        mu, logsigma = self.mu_layer(y), self.sigma_layer(y)
        return mu, logsigma

    def reparameterize(
        self,
        mu: torch.Tensor,
        logsigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma = torch.exp(logsigma / 2)
        epsilion = torch.normal(
            mean=0.0,
            std=torch.ones_like(sigma),
            out=torch.empty_like(sigma),
        )
        return mu + epsilion * sigma

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.image_size)
        mu, logsigma = self.encode(x)
        y = self.reparameterize(mu, logsigma)
        y = self.decode(y)

        return y, mu, logsigma
