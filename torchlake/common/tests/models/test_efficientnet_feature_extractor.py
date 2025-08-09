import pytest
import torch

from ...models import EfficientNetFeatureExtractor


class TestEfficientNetFeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    @pytest.mark.parametrize(
        "network_name",
        ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"],
    )
    @pytest.mark.parametrize("layer_type", ["block"])
    def test_output_shape(
        self,
        network_name: str,
        layer_type: str,
    ):
        self.setUp()
        model = EfficientNetFeatureExtractor(network_name, layer_type=layer_type)
        y = model.forward(
            self.x,
            ["0_1", "1_1", "2_1", "3_1", "4_1", "5_1", "6_1", "7_1", "8_1", "output"],
        )

        for dim, scale in zip(
            model.feature_dims,
            [112, 112, 56, 28, 14, 14, 7, 7, 7],
        ):
            assert y.pop(0).shape == torch.Size((1, dim, scale, scale))

        assert y.pop(0).shape == torch.Size((1, model.feature_dims[-1]))
