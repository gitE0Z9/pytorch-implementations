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
        """A'trous spatial pyramid pooling in paper [1606.00915v2]

        Args:
            input_channel (int): input channel size
            hidden_dim (int): dimension of hidden layer
            output_channel (int): output channel size
            dilations (list[int], optional): dilation size of ASPP, for ASPP-S, it is [2,4,8,12], ASPP-L is default value. Defaults to [6, 12, 18, 24].
        """
        super().__init__()
        self.blocks = nn.ModuleList(
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
        y = self.blocks[0](x)
        for conv in self.blocks[1:]:
            y = y + conv(x)

        return y


class ShallowASPP(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        dilations: list[int] = [],
    ):
        """A'trous spatial pyramid pooling in [deeplabv2.prototxt](http://liangchiehchen.com/projects/DeepLabv2_resnet.html)

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            dilations (list[int], optional): dilation size of ASPP, for ASPP-S, it is [2,4,8,12], ASPP-L is default value. Defaults to [6, 12, 18, 24].
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv2dNormActivation(
                    input_channel,
                    output_channel,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=None,
                    activation_layer=None,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks[0](x)
        for conv in self.blocks[1:]:
            y = y + conv(x)

        return y
