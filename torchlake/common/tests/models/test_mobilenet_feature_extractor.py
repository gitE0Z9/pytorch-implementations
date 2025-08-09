import pytest
import torch

from ...models import MobileNetFeatureExtractor


class TestMobileNetFeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    @pytest.mark.parametrize(
        "network_name,expected_dim,expected_scale",
        [
            ["mobilenet_v2", [32, 24, 32, 64, 160, 1280], [112, 56, 28, 14, 7]],
            ["mobilenet_v3_small", [16, 16, 24, 40, 96, 576], [112, 56, 28, 14, 7]],
            ["mobilenet_v3_large", [16, 24, 40, 80, 160, 960], [112, 56, 28, 14, 7]],
        ],
    )
    def test_output_shape(
        self,
        network_name: str,
        expected_dim: list[int],
        expected_scale: list[int],
    ):
        self.setUp()
        model = MobileNetFeatureExtractor(network_name, layer_type="block")
        y = model.forward(self.x, ["0_1", "1_1", "2_1", "3_1", "4_1", "output"])

        for ele, dim, scale in zip(y[:-1], expected_dim, expected_scale):
            assert ele.shape == torch.Size((1, dim, scale, scale))

        assert y.pop().shape == torch.Size((1, expected_dim[-1]))
