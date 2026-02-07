import torch

from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor

from ..models.refinenet.model import RefineNet
from ..models.refinenet.network import (
    RefineNetBlock,
    MultiResolutionFusion,
    ChainedResidualPooling,
    RCU,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE = 16
NUM_CLASS = 21


class TestNetwork:
    def test_rcu_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = RCU(HIDDEN_DIM)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_multi_resolution_fusion_forward_shape(self):
        shallow_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        deep_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )

        model = MultiResolutionFusion(HIDDEN_DIM + 1, HIDDEN_DIM + 2, HIDDEN_DIM)
        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_chained_residual_pooling_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = ChainedResidualPooling(HIDDEN_DIM, 4)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_refinenet_block_forward_shape(self):
        shallow_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        deep_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )

        model = RefineNetBlock(HIDDEN_DIM + 1, HIDDEN_DIM + 2, HIDDEN_DIM)
        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    def test_refinenet_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet18", trainable=False)
        backbone.fix_target_layers(("1_1", "2_1", "3_1", "4_1"))
        model = RefineNet(backbone, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_refinenet_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet18", trainable=False)
        backbone.fix_target_layers(("1_1", "2_1", "3_1", "4_1"))
        model = RefineNet(backbone, NUM_CLASS)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
