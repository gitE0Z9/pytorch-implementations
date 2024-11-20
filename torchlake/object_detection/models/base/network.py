import torch
from torch import nn


class RegHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        num_priors: int,
        coord_dims: int = 4,
    ):
        """_summary_

        Args:
            input_channel (int): input channel
            num_classes (int): number of classes
            num_priors (int): number of prior boxes
            coord_dims (int, optional): coordinate dimensions. Defaults to 4.
        """
        # mark
        self.num_priors = num_priors
        self.coord_dims = coord_dims
        self.num_classes = num_classes

        super().__init__()
        self.loc = nn.Conv2d(
            input_channel,
            num_priors * coord_dims,
            kernel_size=3,
            padding=1,
        )
        self.conf = nn.Conv2d(
            input_channel,
            num_priors * num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loc(x), self.conf(x)
