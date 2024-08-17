from math import ceil
import pytest
import torch

from ..models.resnet.network import ResBlock
from ..models.senet.model import SeResNet, SeResNeXt
from ..models.senet.network import (
    SeBottleNeck,
    SeConvBlock,
    SeXBottleNeck,
    SeXConvBlock,
)


class TestSeResNet:

    @pytest.mark.parametrize(
        "name,input_channel,base_number,stride",
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
        base_number: int,
        stride: int,
        pre_activation: bool,
    ):
        INPUT_SIZE = 13
        OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

        x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
        layer = SeConvBlock(input_channel, base_number, stride, pre_activation)
        y = layer(x)

        assert y.shape == torch.Size((2, base_number, OUTPUT_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel,stride",
        [
            ["first", 64, 64, 256, 2],
            ["middle", 256, 64, 256, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
        stride: int,
        pre_activation: bool,
    ):
        INPUT_SIZE = 13
        OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

        x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
        # 64 -> 128 -> 128 -> 256
        layer = SeBottleNeck(input_channel, base_number, stride, pre_activation)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel,block,stride",
        [
            ["convnet_first", 64, 128, 128, SeConvBlock, 2],
            ["convnet_middle", 128, 128, 128, SeConvBlock, 1],
            ["bottleneck_first", 64, 64, 256, SeBottleNeck, 2],
            ["bottleneck_middle", 256, 64, 256, SeBottleNeck, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
        block: SeConvBlock | SeBottleNeck,
        stride: int,
        pre_activation: bool,
    ):
        INPUT_SIZE = 13
        OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

        x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
        layer = ResBlock(
            input_channel,
            base_number,
            output_channel,
            block,
            stride,
            pre_activation,
        )
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize("num_layer", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resnet_forward_shape(self, num_layer: int, pre_activation: bool):
        x = torch.randn(2, 3, 224, 224)
        model = SeResNet(
            output_size=5,
            num_layer=num_layer,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))


class TestSeResNeXt:
    @pytest.mark.parametrize(
        "name,input_channel,base_number,stride",
        [
            ["first", 64, 64, 2],
            ["middle", 64, 64, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_convblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        stride: int,
        pre_activation: bool,
    ):
        INPUT_SIZE = 13
        OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

        x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
        layer = SeXConvBlock(input_channel, base_number, stride, pre_activation)
        y = layer(x)

        assert y.shape == torch.Size((2, base_number, OUTPUT_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel,stride",
        [
            ["first", 64, 128, 256, 2],
            ["middle", 256, 128, 256, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
        stride: int,
        pre_activation: bool,
    ):
        INPUT_SIZE = 13
        OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

        x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
        layer = SeXBottleNeck(input_channel, base_number, stride, pre_activation)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel,block,stride",
        [
            ["convnet_first", 64, 64, 64, SeXConvBlock, 2],
            ["bottleneck_first", 64, 128, 256, SeXBottleNeck, 2],
            ["bottleneck_middle", 256, 128, 256, SeXBottleNeck, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
        block: SeXConvBlock | SeXBottleNeck,
        stride: int,
        pre_activation: bool,
    ):
        INPUT_SIZE = 13
        OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

        x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
        layer = ResBlock(
            input_channel,
            base_number,
            output_channel,
            block,
            stride,
            pre_activation,
        )
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize("num_layer", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resnext_forward_shape(self, num_layer: int, pre_activation: bool):
        x = torch.randn(2, 3, 224, 224)
        model = SeResNeXt(
            output_size=5,
            num_layer=num_layer,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))
