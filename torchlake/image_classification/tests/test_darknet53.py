import torch
from torch.testing import assert_close

# from ..models.darknet53.feature_extractor import DarkNet53FeatureExtractor
from ..models.darknet53.model import DarkNet53
from ..models.darknet53.network import darknet_block, darknet_bottleneck


class TestNetwork:
    def test_forward_shape_darknet_block(self):
        x = torch.rand(2, 4, 16, 16)
        layer = darknet_block(4, 8)

        output: torch.Tensor = layer(x)
        assert_close(output.shape, torch.Size([2, 16, 16, 16]))

    def test_forward_shape_darknet_bottleneck(self):
        x = torch.rand(2, 4, 16, 16)
        layer = darknet_bottleneck(4, 8)

        output: torch.Tensor = layer(x)
        assert_close(output.shape, torch.Size([2, 16, 8, 8]))


class TestDarkNet53:
    def test_forward_shape(self):
        x = torch.rand(2, 3, 256, 256)
        model = DarkNet53(output_size=1000)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, 1000]))


# class TestDarkNet53FeatureExtractor:
#     def setUp(self):
#         self.x = torch.rand(1, 3, 256, 256)

#     def test_output_shape(self):
#         self.setUp()

#         model = DarkNet53FeatureExtractor("block")

#         y: torch.Tensor = model.forward(
#             self.x,
#             ["0_1", "1_1", "2_1", "3_1", "4_1", "output"],
#         )
#         for dim, scale in zip(
#             model.feature_dims,
#             [64, 32, 16, 8, 8],
#         ):
#             assert y.pop(0).shape == torch.Size((1, dim, scale, scale))

#         assert y.pop().shape == torch.Size((1, 1024))
