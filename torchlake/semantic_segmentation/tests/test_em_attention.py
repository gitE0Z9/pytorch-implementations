import pytest
import torch

from torchlake.semantic_segmentation.models.deeplabv2.network import (
    deeplab_v2_style_resnet,
)

from ..models.em_attention import EMANet
from ..models.em_attention.network import EMAttention2d


BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = IMAGE_SIZE // 8
NUM_CLASS = 21


class TestNetwork:
    @pytest.mark.parametrize("output_attention", (True, False))
    def test_em_attention_2d_forward_shape(self, output_attention: bool):
        K = 64
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = EMAttention2d(HIDDEN_DIM, k=K)

        y = model(x, output_attention=output_attention)
        if output_attention:
            y, a = y
            assert a.shape == torch.Size((BATCH_SIZE, FEATURE_MAP_SIZE**2, K))

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("output_attention", (True, False))
    def test_emanet_forward_shape(self, output_attention: bool):
        K = 64
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        backbone.fix_target_layers(("4_1",))
        model = EMANet(backbone, NUM_CLASS, k=K)

        y = model(x, output_attention=output_attention)
        if output_attention:
            y, a = y
            assert a.shape == torch.Size((BATCH_SIZE, FEATURE_MAP_SIZE**2, K))

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
