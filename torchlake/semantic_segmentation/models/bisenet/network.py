import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation
from torchlake.common.models.residual import ResBlock
from torchlake.common.models.se import SqueezeExcitation2d


class FeatureFusionModule(nn.Module):
    def __init__(self, input_channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2dNormActivation(input_channel, input_channel, 1),
            ResBlock(
                input_channel,
                input_channel,
                block=SqueezeExcitation2d(input_channel),
            ),
        )

    def forward(self, shallow_x: torch.Tensor, deep_x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            (
                shallow_x,
                F.interpolate(
                    deep_x,
                    size=shallow_x.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ),
            1,
        )
        return self.layers(y)


class AttentionRefinementModule(nn.Module):
    def __init__(self, input_channel: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(input_channel, input_channel, 1),
            nn.BatchNorm2d(input_channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) * x
