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
        "name,input_channel,base_number,output_channel",
        [
            ["first", 64, 64, 256],
            ["middle", 256, 64, 256],
        ],
    )
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = SeBottleNeck(input_channel, base_number)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, 13, 13))

    @pytest.mark.parametrize(
        "name,input_channel,base_number",
        [
            ["first", 64, 128],
            ["middle", 128, 128],
        ],
    )
    def test_convblock_forward_shape(
        self, name: str, input_channel: int, base_number: int
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = SeConvBlock(input_channel, base_number)
        y = layer(x)

        assert y.shape == torch.Size((2, base_number, 13, 13))

    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel,block",
        [
            ["convnet_first", 64, 128, 128, SeConvBlock],
            ["convnet_middle", 128, 128, 128, SeConvBlock],
            ["bottleneck_first", 64, 64, 256, SeBottleNeck],
            ["bottleneck_middle", 256, 64, 256, SeBottleNeck],
        ],
    )
    def test_resblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
        block: SeConvBlock | SeBottleNeck,
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = ResBlock(input_channel, base_number, output_channel, block)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, 13, 13))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["18", 18],
            ["34", 34],
            ["50", 50],
            ["101", 101],
            ["152", 152],
        ],
    )
    def test_resnet_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = SeResNet(output_size=5, num_layer=num_layer)
        y = model(x)

        assert y.shape == torch.Size((2, 5))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["18", 18],
            ["34", 34],
            ["50", 50],
            ["101", 101],
            ["152", 152],
        ],
    )
    def test_resnet2_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = SeResNet(output_size=5, num_layer=num_layer, pre_activation=True)
        y = model(x)

        assert y.shape == torch.Size((2, 5))


class TestSeResNeXt:
    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel",
        [
            ["first", 64, 128, 256],
            ["middle", 256, 128, 256],
        ],
    )
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = SeXBottleNeck(input_channel, base_number)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, 13, 13))

    @pytest.mark.parametrize(
        "name,input_channel,base_number",
        [
            ["first", 64, 64],
        ],
    )
    def test_convblock_forward_shape(
        self, name: str, input_channel: int, base_number: int
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = SeXConvBlock(input_channel, base_number)
        y = layer(x)

        assert y.shape == torch.Size((2, base_number, 13, 13))

    @pytest.mark.parametrize(
        "name,input_channel,base_number,output_channel,block",
        [
            ["convnet_first", 64, 64, 64, SeXConvBlock],
            ["bottleneck_first", 64, 128, 256, SeXBottleNeck],
            ["bottleneck_middle", 256, 128, 256, SeXBottleNeck],
        ],
    )
    def test_resblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        base_number: int,
        output_channel: int,
        block: SeXConvBlock | SeXBottleNeck,
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = ResBlock(input_channel, base_number, output_channel, block)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, 13, 13))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["18", 18],
            ["34", 34],
            ["50", 50],
            ["101", 101],
            ["152", 152],
        ],
    )
    def test_resnet_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = SeResNeXt(output_size=5, num_layer=num_layer)
        y = model(x)

        assert y.shape == torch.Size((2, 5))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["18", 18],
            ["34", 34],
            ["50", 50],
            ["101", 101],
            ["152", 152],
        ],
    )
    def test_resnet2_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = SeResNeXt(output_size=5, num_layer=num_layer, pre_activation=True)
        y = model(x)

        assert y.shape == torch.Size((2, 5))