import torch
from torch import nn


class RegHead(nn.Module):
    def __init__(self, input_channel: int, num_classes: int, num_priors: int):
        super(RegHead, self).__init__()
        self.loc = nn.Conv2d(
            input_channel,
            num_priors * 4,
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
