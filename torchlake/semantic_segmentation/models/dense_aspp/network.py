import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class DenseASPP(nn.Module):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        dilations: list[int],
    ):
        """Dense A'trous spatial pyramid pooling in the paper [DenseASPP for Semantic Segmentation in Street Scenes]

        Args:
            input_channel (int): input channel size of dilated conlutions
            hidden_dim (int): output channel size of dilated conlutions
            dilations (list[int]): dilation size of ASPP, for 16x [6, 12, 18], for 8x [12, 24, 36].
        """
        self.dilations = dilations
        super().__init__()
        self.blocks = nn.ModuleList()
        in_c = input_channel
        for dilation in dilations:
            layer = Conv2dNormActivation(
                in_c,
                hidden_dim,
                3,
                padding=dilation,
                dilation=dilation,
            )
            in_c += hidden_dim
            self.blocks.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = torch.cat((block(x), x), 1)

        return x
