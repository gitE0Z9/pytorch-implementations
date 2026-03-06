import pytest
import torch

from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor

from ..models.bisenet.model import BiSeNet
from ..models.bisenet.network import FeatureFusionModule, AttentionRefinementModule

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 320
HIDDEN_DIM = 8
FEATURE_MAP_SIZE = IMAGE_SIZE // 8
NUM_CLASS = 21


class TestNetwork:
    def test_attention_refinement_module_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))
        model = AttentionRefinementModule(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_feature_fusion_module_forward_shape(self):
        shallow_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        deep_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        model = FeatureFusionModule(HIDDEN_DIM * 2 + 2)

        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM * 2 + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize(
        "is_training,output_shape",
        (
            (True, (IMAGE_SIZE // 8, IMAGE_SIZE // 8)),
            (False, (IMAGE_SIZE, IMAGE_SIZE)),
        ),
    )
    def test_bisenet_forward_shape(self, is_training: bool, output_shape: torch.Size):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet18", trainable=False)
        backbone.fix_target_layers(("3_1", "4_1", "output"))
        model = BiSeNet(backbone, NUM_CLASS)

        if is_training:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, *output_shape))
