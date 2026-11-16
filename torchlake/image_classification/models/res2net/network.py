import torch
from torch import nn

from torchlake.common.models import ConvBNReLU

from ..resnet.network import BottleNeck as BottleNeckBase


class Res2NetLayer(nn.Module):

    def __init__(
        self,
        input_channel: int,
        split: int = 4,
        groups: int = 1,
        pre_activation: bool = False,
    ):
        """Res2Net layer

        Args:
            input_channel (int): input channel size
            split (int, optional): s in paper, split input_channel into s * w channels. Defaults to 4.
            groups (int, optional): group of 3x3 filters. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
        """
        super().__init__()
        assert split >= 2, "split should be at least 2"

        self.split = split
        self.convs = nn.ModuleList(
            [
                ConvBNReLU(
                    input_channel // split,
                    input_channel // split,
                    3,
                    padding=1,
                    group=groups,
                    conv_last=pre_activation,
                )
                for _ in range(split - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.chunk(self.split, 1)
        y = features[0]

        next_y = self.convs[0](features[1])
        y = torch.cat([y, next_y], 1)
        for conv, splited_feature in zip(self.convs[1:], features[2:]):
            next_y = conv(next_y + splited_feature)
            y = torch.cat([y, next_y], 1)

        return y


class BottleNeck(BottleNeckBase):

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        stride: int = 1,
        expansion_factor: int = 4,
        pre_activation: bool = False,
        split: int = 4,
        groups: int = 1,
    ):
        """Bottleneck block in res2net [1904.01169v3]

        kernel: 1x1 -> (3x3 -> 3x3 -> 3x3) -> 1x1
        channel: input_channel -> hidden_dim -> expansion_factor * hidden_dim

        Args:
            input_channel (int): input channel size
            hidden_dim (int): hidden dimension
            stride (int, optional): stride of block. Defaults to 1.
            expansion_factor (int, optional): output channel size = expansion factor x hidden dimension. Defaults to 4.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            split (int, optional): s in paper, split input_channel into s * w channels. Defaults to 4.
            groups (int, optional): group of 3x3 filters. Defaults to 1.
        """
        super().__init__(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        self.layers[1] = Res2NetLayer(hidden_dim, split, groups, pre_activation)
