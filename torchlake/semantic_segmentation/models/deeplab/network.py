from typing import Literal

from torch import nn

from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor


def deeplab_style_vgg(
    network_name: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
    trainable: bool = True,
    large_fov: bool = True,
) -> VGGFeatureExtractor:
    """build backbone with DeepLab style

    Args:
        network_name (Literal["vgg11", "vgg13", "vgg16", "vgg19"]): torchvision vgg model
        trainable (bool, optional): froze the backbone or not. Defaults to False.

    Returns:
        VGGFeatureExtractor: feature extractor
    """
    fe = VGGFeatureExtractor(
        network_name,
        "maxpool",
        trainable,
        enable_gap=False,
        enable_fc1=True,
        enable_fc2=True,
        convert_fc_to_conv=True,
    )
    # stage 5 convs
    for i in range(1, 4):
        fe.feature_extractor[-5 - i * 2].dilation = (2, 2)
        fe.feature_extractor[-5 - i * 2].padding = (2, 2)

    # skip subsampling and keep 8x
    stage = 0
    for layer in fe.feature_extractor:
        if isinstance(layer, nn.MaxPool2d):
            stage += 1
            layer.padding = (1, 1)
            layer.kernel_size = (3, 3)

            # stage 4, 5
            if stage >= 4:
                layer.stride = (1, 1)

    # head
    n = len(fe.feature_extractor)

    fc1_layer: nn.Conv2d = fe.feature_extractor[n - 4]
    if large_fov:
        layer = nn.Conv2d(
            fe.hidden_dim_32x,
            1024,
            3,
            padding=12,
            dilation=12,
        )
        # kernel and dim downsampled
        layer.weight.data.copy_(fc1_layer.weight.data[:1024, :, :3, :3])
        layer.bias.data.copy_(fc1_layer.bias.data[:1024])
        fe.feature_extractor[n - 4] = layer
        fe.feature_dim = 1024

        fc2_layer: nn.Conv2d = fe.feature_extractor[n - 2]
        layer = nn.Conv2d(fe.feature_dim, fe.feature_dim, 1)
        layer.weight.data.copy_(fc2_layer.weight.data[:1024, :1024])
        layer.bias.data.copy_(fc2_layer.bias.data[:1024])
        fe.feature_extractor[n - 2] = layer
    else:
        layer = nn.Conv2d(
            fe.hidden_dim_32x,
            4096,
            4,
            padding=6,
            dilation=4,
        )
        # simple decimation
        layer.weight.data.copy_(fc1_layer.weight.data[:, :, ::2, ::2])
        layer.bias.data.copy_(fc1_layer.bias.data)
        fe.feature_extractor[n - 4] = layer

    fe.feature_extractor.insert(n, nn.Dropout(p=0.5))
    fe.feature_extractor.insert(n - 2, nn.Dropout(p=0.5))
    # XXX: for feature extractor interface compatibility, waste computation
    fe.feature_extractor.append(nn.MaxPool2d(1, 1))

    return fe
