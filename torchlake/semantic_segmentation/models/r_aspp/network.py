from torch import nn

from torchlake.common.models.mobilenet_feature_extractor import (
    MobileNetFeatureExtractor,
)
from torchlake.common.types import MOBILENET_NAMES


def r_aspp_style_mobilenet(
    network_name: MOBILENET_NAMES,
    trainable: bool = True,
) -> MobileNetFeatureExtractor:
    """build backbone, we use mobilenet

    Args:
        backbone_name (Literal[ "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"], optional): mobilenet network name.
        trainable (bool, optional): froze the backbone or not. Defaults to False.

    Returns:
        MobileNetFeatureExtractor: feature extractor
    """
    fe = MobileNetFeatureExtractor(network_name, trainable=trainable)
    block_indices = fe.get_stage()

    for index in block_indices[-2]:
        for key, layer in fe.feature_extractor[index].named_modules():
            layer: nn.Conv2d
            if "block.1.0" in key or "conv.1.0" in key:
                kernel = layer.kernel_size[0]
                dilation = 2
                pad = (dilation * (kernel - 1)) // 2

                layer.dilation, layer.padding, layer.stride = (
                    (dilation, dilation),
                    (pad, pad),
                    (1, 1),
                )

    for index in block_indices[-1]:
        for key, layer in fe.feature_extractor[index].named_modules():
            layer: nn.Conv2d
            if "block.1.0" in key or "conv.1.0" in key:
                # o = i - d(k-1) + 2p
                kernel = layer.kernel_size[0]
                dilation = 4
                pad = (dilation * (kernel - 1)) // 2

                layer.dilation, layer.padding, layer.stride = (
                    (dilation, dilation),
                    (pad, pad),  # +0 for v2, +2 for v3-S, +4 for v3-L
                    (1, 1),
                )

    return fe
