import pytest
import torch

from ...models import VGGFeatureExtractor


class TestVGGFeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    @pytest.mark.parametrize(
        "network_name,expected_block_size",
        [
            ["vgg11", [1, 1, 2, 2, 2]],
            ["vgg13", [2, 2, 2, 2, 2]],
            ["vgg16", [2, 2, 3, 3, 3]],
            ["vgg19", [2, 2, 4, 4, 4]],
        ],
    )
    @pytest.mark.parametrize(
        "layer_type",
        ["conv", "relu", "maxpool"],
    )
    def test_output_shape(
        self,
        network_name: str,
        expected_block_size: list[int],
        layer_type: str,
    ):
        self.setUp()
        model = VGGFeatureExtractor(network_name, layer_type=layer_type)
        y = model.forward(self.x, ["1_1", "2_1", "3_1", "4_1", "5_1"])

        for ele, dim, scale in zip(y, [64, 128, 256, 512, 512], [112, 56, 28, 14, 7]):
            if layer_type != "maxpool":
                scale *= 2
            assert ele.shape == torch.Size((1, dim, scale, scale))
