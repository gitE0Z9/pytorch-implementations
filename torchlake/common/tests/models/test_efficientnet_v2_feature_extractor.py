import pytest
import torch

from ...models import EfficientNetV2FeatureExtractor


class TestEfficientNetV2FeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    @pytest.mark.parametrize("network_name", ["s", "m", "l"])
    @pytest.mark.parametrize("layer_type", ["block"])
    def test_output_shape(
        self,
        network_name: str,
        layer_type: str,
    ):
        self.setUp()
        model = EfficientNetV2FeatureExtractor(network_name, layer_type=layer_type)
        layers = [
            "0_1",
            "1_1",
            "2_1",
            "3_1",
            "4_1",
            "5_1",
            "6_1",
            "7_1",
            "8_1",
            "output",
        ]
        expected_scales = [112, 112, 56, 28, 14, 14, 7, 7, 7]

        if network_name == "s":
            layers.pop(-2)
            expected_scales.pop(-2)

        y = model.forward(self.x, layers)

        for dim, scale in zip(model.feature_dims, expected_scales):
            assert y.pop(0).shape == torch.Size((1, dim, scale, scale))

        assert y.pop(0).shape == torch.Size((1, model.feature_dims[-1]))
