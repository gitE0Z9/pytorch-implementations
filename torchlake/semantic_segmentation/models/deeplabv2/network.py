from typing import Literal
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor
from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor
from torchlake.common.types import RESNET_NAMES, VGG_NAMES
from ..deeplab.network import deeplab_style_vgg


def deeplab_v2_style_vgg(
    network_name: VGG_NAMES,
    trainable: bool = True,
    large_fov: bool = True,
    pool_5_type: Literal["max", "avg"] = "max",
) -> VGGFeatureExtractor:
    """build backbone with DeepLab style

    Args:
        network_name (Literal["vgg11", "vgg13", "vgg16", "vgg19"]): torchvision vgg model
        trainable (bool, optional): froze the backbone or not. Defaults to False.
        large_fov (bool, optional): use larger field of view. Defaults to True.
        pool_5_type (Literal["max", "avg"], optional): the type of pool 5. Defaults to "max".

    Returns:
        VGGFeatureExtractor: feature extractor
    """
    fe = deeplab_style_vgg(
        network_name,
        trainable,
        large_fov,
        pool_5_type,
    )

    return fe


def deeplab_v2_style_resnet(
    network_name: RESNET_NAMES,
    trainable: bool = True,
    dilation_size_16x: int = 2,
    dilation_size_32x: int = 4,
) -> ResNetFeatureExtractor:
    fe = ResNetFeatureExtractor(network_name, trainable=trainable, enable_gap=False)

    for key, layer in fe.feature_extractor[6].named_modules():
        layer: nn.Conv2d
        if "conv2" in key:
            layer.dilation, layer.padding, layer.stride = (
                (dilation_size_16x, dilation_size_16x),
                (dilation_size_16x, dilation_size_16x),
                (1, 1),
            )
        elif "downsample.0" in key:
            layer.stride = (1, 1)

    for key, layer in fe.feature_extractor[7].named_modules():
        if "conv2" in key:
            layer.dilation, layer.padding, layer.stride = (
                (dilation_size_32x, dilation_size_32x),
                (dilation_size_32x, dilation_size_32x),
                (1, 1),
            )
        elif "downsample.0" in key:
            layer.stride = (1, 1)

    return fe


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
        self.blocks = nn.ModuleList([])
        for dilation in dilations:
            layer = nn.Sequential(
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

            nn.init.normal_(layer[4][0].weight.data, 0, 0.01)
            nn.init.zeros_(layer[4][0].bias.data)

            self.blocks.append(layer)

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
        self.blocks = nn.ModuleList([])

        for dilation in dilations:
            layer = Conv2dNormActivation(
                input_channel,
                output_channel,
                3,
                padding=dilation,
                dilation=dilation,
                norm_layer=None,
                activation_layer=None,
            )
            nn.init.normal_(layer[0].weight.data, 0, 0.01)
            nn.init.zeros_(layer[0].bias.data)

            self.blocks.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks[0](x)
        for conv in self.blocks[1:]:
            y = y + conv(x)

        return y
