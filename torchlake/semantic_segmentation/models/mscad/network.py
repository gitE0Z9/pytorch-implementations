from itertools import pairwise

from torch import nn
from torchlake.common.models.cnn_base import ModelBase
from torchvision.ops import Conv2dNormActivation


class ContextModule(ModelBase):

    def __init__(
        self,
        input_channel: int,
        channel_ratios: list[int],
        dilations: list[int] = [1, 1, 2, 4, 8, 16, 1, 1],
    ):
        self.hidden_dim = input_channel
        self.channel_ratios = channel_ratios
        self.dilations = dilations
        super().__init__(input_channel, input_channel)

    def build_foot(self, input_channel: int):
        self.foot = Conv2dNormActivation(
            input_channel,
            self.hidden_dim * self.channel_ratios[0],
            3,
            dilation=self.dilations[0],
            padding=self.dilations[0],
            # padding=0,
            norm_layer=None,
        )

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                Conv2dNormActivation(
                    self.hidden_dim * channel_ratio,
                    self.hidden_dim * channel_ratio_next,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    # padding=0,
                    norm_layer=None,
                )
                for (channel_ratio, channel_ratio_next), dilation in zip(
                    pairwise(self.channel_ratios), self.dilations[1:]
                )
            ]
        )

    def build_head(self, _: int):
        self.head = nn.Identity()


def context_network_basic(input_channel: int):
    return ContextModule(input_channel, [1] * 8)


def context_network_large(input_channel: int):
    return ContextModule(input_channel, [2, 2, 4, 8, 16, 32, 32, 1])
