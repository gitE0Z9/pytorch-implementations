import torch
from torch.testing import assert_close

from ..models.darknet19.feature_extractor import DarkNet19FeatureExtractor
from ..models.darknet19.model import DarkNet19
from ..models.darknet19.network import BottleNeck


class TestBottleNeck:
    def test_output_shape(self):
        """input: 4
        block: 8 -> 4 -> 8
        """
        layer = BottleNeck(8, 3)
        test_x = torch.rand(2, 4, 16, 16)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 8, 16, 16]))


class TestDarknet19:
    def test_output_shape(self):
        model = DarkNet19(output_size=1000)
        x = torch.rand(2, 3, 256, 256)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, 1000]))


class TestDarkNet19FeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 256, 256)

    def test_output_shape(self):
        self.setUp()

        model = DarkNet19FeatureExtractor("block")

        y: torch.Tensor = model.forward(
            self.x,
            ["0_1", "1_1", "2_1", "3_1", "4_1", "output"],
        )
        for dim, scale in zip(
            model.feature_dims,
            [64, 32, 16, 8, 8],
        ):
            assert y.pop(0).shape == torch.Size((1, dim, scale, scale))

        assert y.pop().shape == torch.Size((1, 1024))
