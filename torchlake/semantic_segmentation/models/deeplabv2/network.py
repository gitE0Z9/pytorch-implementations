import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class ASPP(nn.Module):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_channel: int,
        dilations: list[int] = [],
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dNormActivation(
                        input_channel,
                        hidden_dim,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        norm_layer=None,
                    ),
                    nn.Dropout(p=0.5),
                    Conv2dNormActivation(
                        hidden_dim,
                        hidden_dim,
                        1,
                        norm_layer=None,
                    ),
                    nn.Dropout(p=0.5),
                    Conv2dNormActivation(
                        hidden_dim,
                        output_channel,
                        1,
                        norm_layer=None,
                    ),
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convs[0](x)
        for conv in self.convs[1:]:
            y = y + conv(x)

        return y
