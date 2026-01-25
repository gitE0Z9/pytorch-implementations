from itertools import pairwise
from typing import Sequence

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor
from torchlake.common.types import VGG_NAMES


def dilation_net_style_vgg(
    network_name: VGG_NAMES,
    trainable: bool = True,
) -> VGGFeatureExtractor:
    """build backbone with Multi-scale context aggragration by dilation convolution in paper [1511.07122v3]

    Args:
        network_name (Literal["vgg11", "vgg13", "vgg16", "vgg19"]): torchvision vgg model
        trainable (bool, optional): froze the backbone or not. Defaults to False.

    Returns:
        VGGFeatureExtractor: feature extractor
    """
    fe = VGGFeatureExtractor(
        network_name,
        "maxpool",
        trainable=trainable,
        enable_fc1=True,
        enable_fc2=True,
        convert_fc_to_conv=True,
    )

    # stage 5 convs
    for i in range(1, 4):
        fe.feature_extractor[-5 - i * 2].dilation = (2, 2)
        fe.feature_extractor[-5 - i * 2].padding = (2, 2)

    # skip subsampling and keep 8x
    # TODO: remove every padding
    stage = 0
    for layer in fe.feature_extractor:
        if isinstance(layer, nn.MaxPool2d):
            stage += 1

            # stage 4, 5, remove pooling
            if stage >= 4:
                layer.stride, layer.kernel_size = (1, 1), (1, 1)

        # stage other & 5 convs
        # if isinstance(layer, nn.Conv2d):
        # layer.padding = (0, 0)

    # head
    n = len(fe.feature_extractor)

    fe.feature_extractor[n - 4].dilation = (4, 4)
    fe.feature_extractor[n - 4].padding = (12, 12)

    fe.feature_extractor.insert(n, nn.Dropout(p=0.5))
    fe.feature_extractor.insert(n - 2, nn.Dropout(p=0.5))
    # XXX: for feature extractor interface compatibility, waste computation
    fe.feature_extractor.append(nn.MaxPool2d(1, 1))

    return fe


class ContextModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        multipliers: Sequence[int],
        dilations: Sequence[int] = (1, 1, 2, 4, 8, 16, 1, 1),
        epsilon: float = 1e-2,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.multipliers = multipliers
        self.dilations = dilations
        self.epsilon = epsilon
        assert len(multipliers) == len(
            dilations
        ), "the number of channel ratios should be the same as the number of dilations"

        self.blocks = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                input_channel * self.multipliers[0],
                3,
                dilation=self.dilations[0],
                padding=self.dilations[0],
                norm_layer=None,
            ),
            *[
                Conv2dNormActivation(
                    input_channel * multiplier,
                    input_channel * multiplier_next,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_layer=None,
                )
                for (multiplier, multiplier_next), dilation in zip(
                    pairwise(self.multipliers), self.dilations[1:]
                )
            ],
            Conv2dNormActivation(
                input_channel * self.multipliers[-1],
                input_channel,
                1,
                norm_layer=None,
                activation_layer=None,
            ),
        )

        self._init_identity_weights()

    def _init_identity_weights(self):
        """initialize context module with identity
        [source](https://github.com/fyu/caffe-dilation/blob/9548b9a44c49986f0fb5a73447e59b58c187143f/include/caffe/filler.hpp#L267)
        """
        for i, multiplier in enumerate(self.multipliers):
            layer = self.blocks[i][0]
            out_c, in_c, h, w = layer.weight.data.shape

            # Fill with small noise
            nn.init.normal_(layer.weight.data, 0, self.epsilon / multiplier)
            nn.init.zeros_(layer.bias.data)

            # Set identity at center
            center_h, center_w = h // 2, w // 2
            num_groups = in_c
            out_group_size = out_c // num_groups

            for g in range(num_groups):
                out_start = g * out_group_size
                out_end = (g + 1) * out_group_size
                layer.weight.data[out_start:out_end, g, center_h, center_w] = (
                    1.0 / multiplier
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def context_module_basic(input_channel: int):
    return ContextModule(input_channel, [1] * 8)


def context_module_large(input_channel: int):
    return ContextModule(input_channel, [2, 2, 4, 8, 16, 32, 32, 1])
