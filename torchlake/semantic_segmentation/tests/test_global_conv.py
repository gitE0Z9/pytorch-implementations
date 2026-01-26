import torch

from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor

from ..models.global_conv.model import GlobalConvolutionNetwork
from ..models.global_conv.network import BoundaryRefinement, GlobalConvolutionBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE = 16
NUM_CLASS = 21


class TestNetwork:
    def test_global_convolution_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, NUM_CLASS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = GlobalConvolutionBlock(NUM_CLASS, NUM_CLASS, 5)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_CLASS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_boundary_refinement_forward_shape(self):
        x = torch.rand((BATCH_SIZE, NUM_CLASS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = BoundaryRefinement(NUM_CLASS)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_CLASS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    def test_global_convolution_network_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet50", trainable=False)
        model = GlobalConvolutionNetwork(backbone, NUM_CLASS, kernel=15)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
