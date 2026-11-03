from math import ceil

import pytest
import torch

from ..models.resnet.model import ResNet, ResNetB, ResNetC, ResNetD
from ..models.resnet.network import (
    BottleNeck,
    ConvBlock,
    BottleNeckB,
    ResBlockD,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 13


class TestNetwork:
    @pytest.mark.parametrize(
        "name,input_channel,output_channel,stride",
        [
            ["first", 64, 128, 2],
            ["middle", 128, 128, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_convblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        stride: int,
        pre_activation: bool,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        model = ConvBlock(input_channel, output_channel, stride, pre_activation)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SIZE, OUTPUT_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,hidden_dim,expansion_factor,stride",
        [
            ["first", 64, 64, 4, 2],
            ["middle", 256, 64, 4, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    @pytest.mark.parametrize("block", [BottleNeck, BottleNeckB])
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        hidden_dim: int,
        expansion_factor: int,
        stride: int,
        pre_activation: bool,
        block: BottleNeck | BottleNeckB,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        model = block(
            input_channel,
            hidden_dim,
            stride=stride,
            pre_activation=pre_activation,
            expansion_factor=expansion_factor,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, hidden_dim * expansion_factor, OUTPUT_SIZE, OUTPUT_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,hidden_dim,expansion_factor,stride",
        [
            ["bottleneck_first", 64, 64, 4, 2],
            ["bottleneck_middle", 256, 64, 4, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resblock_d_forward_shape(
        self,
        name: str,
        input_channel: int,
        hidden_dim: int,
        expansion_factor: int,
        stride: int,
        pre_activation: bool,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        model = ResBlockD(
            input_channel,
            hidden_dim * expansion_factor,
            block=BottleNeckB(
                input_channel,
                hidden_dim,
                stride=stride,
                expansion_factor=expansion_factor,
                pre_activation=pre_activation,
            ),
            stride=stride,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, hidden_dim * expansion_factor, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    @pytest.mark.parametrize("model_cls", [ResNet, ResNetB, ResNetC, ResNetD])
    def test_resnet_forward_shape(
        self,
        key: int,
        pre_activation: bool,
        model_cls: ResNet | ResNetB | ResNetC | ResNetD,
    ):
        if key in [18, 34] and model_cls != ResNet:
            pytest.skip("version B, C, D only work with bottleneck layer.")

        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = model_cls(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
