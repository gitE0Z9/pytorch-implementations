import pytest
import torch
from torch import nn

from ...models import VGGFeatureExtractor

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224


class TestVGGFeatureExtractor:
    @pytest.mark.parametrize("network_name", ["vgg11", "vgg13", "vgg16", "vgg19"])
    @pytest.mark.parametrize("enable_bn", [True, False])
    @pytest.mark.parametrize(
        "layer_type",
        ["conv", "relu", "maxpool"],
    )
    def test_output_shape(
        self,
        network_name: str,
        enable_bn: bool,
        layer_type: str,
    ):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = VGGFeatureExtractor(
            network_name,
            layer_type=layer_type,
            enable_bn=enable_bn,
        )
        y = model.forward(x, ["1_1", "2_1", "3_1", "4_1", "5_1"])

        for ele, dim, scale in zip(y, [64, 128, 256, 512, 512], [112, 56, 28, 14, 7]):
            if layer_type != "maxpool":
                scale *= 2
            assert ele.shape == torch.Size((BATCH_SIZE, dim, scale, scale))

    @pytest.mark.parametrize("network_name", ["vgg11", "vgg13", "vgg16", "vgg19"])
    @pytest.mark.parametrize("enable_bn", [True, False])
    def test_get_stage(self, network_name, enable_bn):
        model = VGGFeatureExtractor(
            network_name,
            layer_type="conv",
            enable_bn=enable_bn,
        )

        stages = model.get_stage()
        for stage in stages:
            for block_index in stage:
                assert isinstance(model.feature_extractor[block_index], nn.Conv2d)
