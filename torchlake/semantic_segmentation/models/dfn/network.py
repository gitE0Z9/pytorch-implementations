import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.residual import ResBlock


class RefinementResidualBlock(nn.Module):

    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1),
            ResBlock(
                output_channel,
                output_channel,
                block=nn.Sequential(
                    Conv2dNormActivation(output_channel, output_channel, 3),
                    nn.Conv2d(output_channel, output_channel, 3, padding=1),
                ),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ChannelAttentionBlock(nn.Module):

    def __init__(
        self,
        shallow_input_channel: int,
        deep_input_channel: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(shallow_input_channel * 2, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, shallow_input_channel, 1),
            nn.Sigmoid(),
        )

        if shallow_input_channel != deep_input_channel:
            self.branch_deep = nn.Sequential(
                nn.Conv2d(deep_input_channel, shallow_input_channel, 1),
            )

    def forward(self, shallow_x: torch.Tensor, deep_x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "branch_deep"):
            deep_y = self.branch_deep(deep_x)
        else:
            deep_y = deep_x
        deep_y = F.interpolate(
            deep_y,
            size=shallow_x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        y = torch.cat((shallow_x, deep_y), 1)
        y = self.se(y) * shallow_x
        return y + deep_y
