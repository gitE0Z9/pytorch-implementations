import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.residual import ResBlock


class GlobalConvolutionBlock(nn.Module):

    def __init__(self, input_channel: int, output_size: int, kernel: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_channel,
                        output_size,
                        (kernel, 1),
                        padding=(kernel // 2, 0),
                    ),
                    nn.Conv2d(
                        output_size,
                        output_size,
                        (1, kernel),
                        padding=(0, kernel // 2),
                    ),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        input_channel,
                        output_size,
                        (1, kernel),
                        padding=(0, kernel // 2),
                    ),
                    nn.Conv2d(
                        output_size,
                        output_size,
                        (kernel, 1),
                        padding=(kernel // 2, 0),
                    ),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branches[0](x) + self.branches[1](x)


class BoundaryRefinement(nn.Module):

    def __init__(self, output_size: int):
        super().__init__()
        self.blocks = ResBlock(
            output_size,
            output_size,
            block=nn.Sequential(
                Conv2dNormActivation(output_size, output_size, 3, norm_layer=None),
                nn.Conv2d(output_size, output_size, 3, padding=1),
            ),
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
