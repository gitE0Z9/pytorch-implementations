import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class Res2NetLayer(nn.Module):

    def __init__(
        self,
        input_channel: int,
        split: int = 4,
        groups: int = 1,
    ):
        """Res2Net module

        Args:
            input_channel (int): input channel size
            split (int, optional): s in paper, split input_channel into s * w channels. Defaults to 4.
            groups (int, optional): group of 3x3 filters. Defaults to 1.
        """
        super(Res2NetLayer, self).__init__()
        self.split = split
        self.convs = nn.ModuleList(
            [
                Conv2dNormActivation(
                    input_channel // split,
                    input_channel // split,
                    3,
                    groups=groups,
                )
            ]
            * (split - 1)
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


class BottleNeck(nn.Module):

    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        stride: int = 1,
        split: int = 4,
        groups: int = 1,
    ):
        """Bottleneck block in res2net [1904.01169v3]
        The modified middle portion is a VGG extractor liked structure.

        1 -> (3 -> 3 -> 3) -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            stride (int, optional): stride of block. Defaults to 1.
            split (int, optional): s in paper, split input_channel into s * w channels. Defaults to 4.
            groups (int, optional): group of 3x3 filters. Defaults to 1.
        """
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                block_base_channel,
                1,
                stride=stride,
            ),
            Res2NetLayer(
                block_base_channel,
                split,
                groups,
            ),
            Conv2dNormActivation(
                block_base_channel,
                block_base_channel * 4,
                1,
                activation_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
