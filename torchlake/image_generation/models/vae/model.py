import torch
from torch import nn
from torchlake.common.models.model_base import ModelBase


class VAE(ModelBase):
    def __init__(
        self,
        input_channel: int = 1,
        image_size: int = 28 * 28,
        hidden_dim: int = 128,
        latent_dim: int = 64,
    ):
        self.image_size = image_size
        self.total_pixel = input_channel * image_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        super().__init__(self.total_pixel, self.total_pixel)

    def build_foot(self, input_channel):
        self.foot = nn.Sequential(
            nn.Linear(input_channel, self.hidden_dim),
            nn.ReLU(),
        )

    def build_neck(self):
        self.neck = nn.ModuleDict(
            {
                "mu": nn.Linear(self.hidden_dim, self.latent_dim),
                "sigma": nn.Linear(self.hidden_dim, self.latent_dim),
            }
        )

    def build_head(self, output_size):
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_size),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.foot(x)
        return self.neck["mu"](y), self.neck["sigma"](y)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(logvar / 2)
        epsilion = torch.normal(
            mean=0.0,
            std=torch.ones_like(sigma),
            out=torch.empty_like(sigma),
        )
        return mu + epsilion * sigma

    def decode(self, x: torch.Tensor, output_shape: torch.Size = None) -> torch.Tensor:
        y = self.head(x)

        if output_shape is not None:
            return y.view(*output_shape)

        return y

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x = x.view(-1, self.total_pixel)

        mu, logvar = self.encode(x)
        y = self.sample(mu, logvar)
        y = self.decode(y, original_shape)

        if self.training:
            return y, mu, logvar

        return y
