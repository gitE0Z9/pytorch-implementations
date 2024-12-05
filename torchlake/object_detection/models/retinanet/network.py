import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class RegHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int = 256,
        num_priors: int = 9,
        num_classes: int = 1,
        coord_dims: int = 4,
    ):
        """_summary_

        Args:
            input_channel (int): input channel
            hidden_dim (int, optional): hidden dimensions. Defaults to 256.
            num_priors (int, optional): number of prior boxes. Defaults to 1.
            num_classes (int, optional): number of classes. Defaults to 1.
            coord_dims (int, optional): coordinate dimensions. Defaults to 4.
        """
        # mark
        self.num_priors = num_priors
        self.num_classes = num_classes
        self.coord_dims = coord_dims

        super().__init__()
        self.loc = nn.Sequential(
            Conv2dNormActivation(input_channel, hidden_dim, 3, norm_layer=None),
            Conv2dNormActivation(hidden_dim, hidden_dim, 3, norm_layer=None),
            Conv2dNormActivation(hidden_dim, hidden_dim, 3, norm_layer=None),
            Conv2dNormActivation(hidden_dim, hidden_dim, 3, norm_layer=None),
            nn.Conv2d(
                hidden_dim,
                num_priors * coord_dims,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conf = nn.Sequential(
            Conv2dNormActivation(input_channel, hidden_dim, 3, norm_layer=None),
            Conv2dNormActivation(hidden_dim, hidden_dim, 3, norm_layer=None),
            Conv2dNormActivation(hidden_dim, hidden_dim, 3, norm_layer=None),
            Conv2dNormActivation(hidden_dim, hidden_dim, 3, norm_layer=None),
            nn.Conv2d(
                hidden_dim,
                num_priors * num_classes,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.loc(x), self.conf(x)], 1)
