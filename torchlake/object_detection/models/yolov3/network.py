import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class RegHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_priors: int,
        num_classes: int,
        coord_dims: int = 4,
    ):
        """_summary_

        Args:
            input_channel (int): input channel
            num_priors (int): number of prior boxes
            num_classes (int): number of classes
            coord_dims (int, optional): coordinate dimensions. Defaults to 4.
        """
        # mark
        self.num_priors = num_priors
        self.coord_dims = coord_dims
        self.num_classes = num_classes

        super().__init__()
        self.layers = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                input_channel * 2,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.Conv2d(
                input_channel * 2,
                num_priors * (num_classes + coord_dims + 1),
                1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
