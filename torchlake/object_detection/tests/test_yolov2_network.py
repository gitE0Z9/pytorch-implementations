import torch
from torch.testing import assert_close

from ..models.yolov2.network import BottleNeck, ConvBlock, Darknet19, ReorgLayer


class TestConvBlock:
    def test_1x1_output_shape(self):
        layer = ConvBlock(3, 16, 1)
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 16, 224, 224]))

    def test_3x3_output_shape(self):
        layer = ConvBlock(3, 16, 3)
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 16, 224, 224]))


class TestBottleNeck:
    def test_output_shape(self):
        """input: 4
        block: 8 -> 4 -> 8
        """
        layer = BottleNeck(8, 3)
        test_x = torch.rand(2, 4, 224, 224)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 8, 224, 224]))


class Test_Reorg:
    def test_output_shape(self):
        layer = ReorgLayer(2)
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 12, 112, 112]))


class TestDarknet19:
    def test_output_shape(self):
        model = Darknet19()
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 1000]))
