import pytest
import torch

from ...models import ResNetFeatureExtractor


class TestResNetFeatureExtractor:
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)

    # @pytest.mark.parametrize(
    #     "network_name,num_layers",
    #     [
    #         ["resnet18", [2, 2, 2, 2]],
    #         ["resnet34", [3, 4, 6, 3]],
    #         ["resnet50", [3, 4, 6, 3]],
    #         ["resnet101", [3, 4, 23, 3]],
    #         ["resnet152", [3, 8, 36, 3]],
    #     ],
    # )
    # def test_backbone(self, network_name: str, num_layers: list[int]):
    #     model = ResNetFeatureExtractor(network_name, layer_type="block")

    #     for block, num_layer in zip(iter(model.feature_extractor), [4, *num_layers, 1]):
    #         # skip avgpool, since no len
    #         if getattr(block, "__len__", None):
    #             assert len(block) == num_layer

    # @pytest.mark.parametrize(
    #     "network_name",
    #     ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    # )
    # def test_equalness(self, network_name: str):
    #     self.setUp()
    #     model = ResNetFeatureExtractor(network_name, layer_type="block")
    #     features = model.forward(self.x, ["0_1", "1_1", "2_1", "3_1", "4_1", "output"])

    #     from torchvision import models

    #     original_model = getattr(models, network_name)(weights="DEFAULT")

    #     y = model.normalization(self.x)
    #     y = original_model.conv1(y)
    #     y = original_model.bn1(y)
    #     y = original_model.relu(y)
    #     y = original_model.maxpool(y)
    #     assert_close(features.pop(0), y)
    #     y = original_model.layer1(y)
    #     assert_close(features.pop(0), y)
    #     y = original_model.layer2(y)
    #     assert_close(features.pop(0), y)
    #     y = original_model.layer3(y)
    #     assert_close(features.pop(0), y)
    #     y = original_model.layer4(y)
    #     assert_close(features.pop(0), y)
    #     y = original_model.avgpool(y)
    #     assert_close(features.pop(0), y.squeeze((-1, -2)))

    @pytest.mark.parametrize("num_layer", [18, 34, 50, 101, 152])
    def test_output_shape(self, num_layer: int):
        self.setUp()
        model = ResNetFeatureExtractor(f"resnet{num_layer}", layer_type="block")
        y = model.forward(self.x, ["0_1", "1_1", "2_1", "3_1", "4_1", "output"])

        factor = 1 if num_layer >= 50 else 4

        for dim, scale in zip(
            [64, 256 // factor, 512 // factor, 1024 // factor, 2048 // factor],
            [56, 56, 28, 14, 7],
        ):
            assert y.pop(0).shape == torch.Size((1, dim, scale, scale))

        assert y.pop().shape == torch.Size((1, 2048 // factor))
