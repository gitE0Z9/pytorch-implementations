from typing import Literal

from torch import nn

from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor
from torchlake.common.utils.numerical import bilinear_kernel


def init_conv_to_zeros(layer: nn.Conv2d):
    nn.init.zeros_(layer.weight.data)
    nn.init.zeros_(layer.bias.data)


def init_deconv_with_bilinear_kernel(layer: nn.ConvTranspose2d):
    out_c, in_c, k, _ = layer.weight.data.shape
    layer.weight.data.copy_(bilinear_kernel(in_c, out_c, k))


def fcn_style_vgg(
    network_name: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
    trainable: bool = True,
) -> VGGFeatureExtractor:
    fe = VGGFeatureExtractor(
        network_name,
        "maxpool",
        trainable,
        enable_gap=False,
        enable_fc1=True,
        enable_fc2=True,
        convert_fc_to_conv=True,
    )
    n = len(fe.feature_extractor)
    fe.feature_extractor.insert(n, nn.Dropout(p=0.5))
    fe.feature_extractor.insert(n - 2, nn.Dropout(p=0.5))
    # XXX: for feature extractor interface compatibility, waste computation
    fe.feature_extractor.append(nn.MaxPool2d(1, 1))

    fe.feature_extractor[0].padding = (100, 100)

    return fe
