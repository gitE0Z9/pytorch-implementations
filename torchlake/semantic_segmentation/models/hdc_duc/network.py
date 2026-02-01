from typing import Sequence

import torch
from torch import nn

from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor
from torchlake.common.types import RESNET_NAMES


def is_hdc_rule_valid(dilations: Sequence[int], kernel: int) -> bool:
    if len(dilations) < 2:
        return True

    dilations = dilations[::-1]
    M_i = dilations[0]
    for i in range(1, len(dilations)):
        r_next, r_i = dilations[i - 1], dilations[i]
        if r_next < r_i:
            M_i = r_i
        else:
            M_i = max(r_i, M_i - 2 * r_i, 2 * r_i - M_i)

        if r_i != M_i:
            return False
        # M_2 == r_2, that is i = 3
        # if i == len(dilations) - 2 and M_i > kernel:
        #     return False

    return True


def tusimple_style_resnet(
    network_name: RESNET_NAMES,
    trainable: bool = True,
    dilation_sizes_16x: Sequence[int] = (2, 2, 5, 9) + 4 * (1, 2, 5, 9) + (1, 2, 5),
    dilation_sizes_32x: Sequence[int] = (5, 9, 17),
    output_stride: int = 8,
) -> ResNetFeatureExtractor:
    """ResNet with HDC

    Args:
        network_name (Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]): torchvision resnet model
        trainable (bool, optional): backbone is trainable or not. Defaults to True.
        dilation_sizes_16x (Sequence[int], optional): dilation sizes of stage 4 of ResNet. Defaults to (2, 2, 5, 9)+4*(1, 2, 5, 9)+(1, 2, 5).
        dilation_sizes_32x (Sequence[int], optional): dilation size of stage 5 of ResNet. Defaults to (5, 9, 17).

    Raises:
        ValueError: when dilation_sizes_16x or dilation_sizes_32x does not match the HDC rule

    Returns:
        ResNetFeatureExtractor: resnet feature extractor
    """
    if not is_hdc_rule_valid(dilation_sizes_16x, 3):
        raise ValueError("dilation_sizes_16x violates DUC rule")
    if not is_hdc_rule_valid(dilation_sizes_32x, 3):
        raise ValueError("dilation_sizes_32x violates DUC rule")

    fe = ResNetFeatureExtractor(network_name, trainable=trainable)

    i = 0
    for key, layer in fe.feature_extractor[6].named_modules():
        layer: nn.Conv2d
        if "conv2" in key:
            d = dilation_sizes_16x[i]
            layer.dilation, layer.padding = ((d, d), (d, d))
            if output_stride <= 8:
                layer.stride = (1, 1)

            i += 1
        elif "downsample.0" in key and output_stride <= 8:
            layer.stride = (1, 1)

    i = 0
    for key, layer in fe.feature_extractor[7].named_modules():
        if "conv2" in key:
            d = dilation_sizes_32x[i]
            layer.dilation, layer.padding = ((d, d), (d, d))
            if output_stride <= 16:
                layer.stride = (1, 1)

            i += 1
        elif "downsample.0" in key and output_stride <= 16:
            layer.stride = (1, 1)

    return fe


class DUC(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
        output_stride: int,
        dilations: Sequence[int] = (6, 12, 18),
    ):
        """Dense Upsampling Convolution in [1702.08502v3]"""
        super().__init__()
        self.output_size = output_size
        self.output_stride = output_stride
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channel,
                    output_size * output_stride**2,
                    3,
                    padding=dilation,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y += layer(x)

        return (
            y.view(b, self.output_size, self.output_stride, self.output_stride, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(
                b, self.output_size, h * self.output_stride, w * self.output_stride
            )
        )
