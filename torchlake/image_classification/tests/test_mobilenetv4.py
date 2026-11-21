from math import ceil
from typing import Sequence

import pytest
import torch

from ..models.mobilenetv4.model import MobileNetV4
from ..models.mobilenetv4.network import (
    FusedInvertedBottleneck,
    LayerScale2d,
    MobileMQA,
    UniversalInvertedBottleneck,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 14
HIDDEN_DIM = 64


class TestNetwork:
    def test_layer_scale_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = LayerScale2d(HIDDEN_DIM)
        y = model(x)

        assert y.shape == x.shape

    @pytest.mark.parametrize(
        "name,input_channel,output_channel,stride",
        [
            ["downsample", 32, 48, 2],
            ["residual", 48, 48, 1],
        ],
    )
    def test_fused_inverted_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        stride: int,
    ):
        OUTPUT_SHAPE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = FusedInvertedBottleneck(input_channel, output_channel, 3, stride)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SHAPE, OUTPUT_SHAPE)
        )

    @pytest.mark.parametrize(
        "name,dw_kernels,input_channel,output_channel,stride",
        [
            ["inverted_bottleneck_downsample", (0, 3), 32, 64, 2],
            ["inverted_bottleneck_residual", (0, 3), 64, 64, 1],
            ["convnext_like_downsample", (3, 0), 32, 64, 2],
            ["convnext_like_residual", (3, 0), 64, 64, 1],
            ["extra_dw_downsample", (3, 5), 32, 64, 2],
            ["extra_dw_residual", (3, 5), 64, 64, 1],
            ["ffn_residual", (0, 0), 64, 64, 1],
        ],
    )
    @pytest.mark.parametrize("enable_layer_scale", [True, False])
    def test_universal_inverted_bottleneck_forward_shape(
        self,
        name: str,
        dw_kernels: Sequence[int],
        input_channel: int,
        output_channel: int,
        stride: int,
        enable_layer_scale: bool,
    ):
        OUTPUT_SHAPE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = UniversalInvertedBottleneck(
            input_channel,
            output_channel,
            dw_kernels,
            stride,
            enable_layer_scale=enable_layer_scale,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SHAPE, OUTPUT_SHAPE)
        )

    @pytest.mark.parametrize("kv_stride", [1, 2])
    @pytest.mark.parametrize("num_head", [1, 4])
    def test_mobile_mqa_forward_shape(self, kv_stride: int, num_head: int):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = MobileMQA(HIDDEN_DIM, num_head, 32, 48, kv_stride)
        y = model(x)

        assert y.shape == x.shape


class TestModel:
    @pytest.mark.parametrize(
        "key",
        [
            "conv_small",
            "conv_medium",
            "conv_large",
            "hybrid_medium",
            "hybrid_large",
        ],
    )
    def test_mobilenetv4_forward_shape(self, key: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = MobileNetV4(INPUT_CHANNEL, OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
