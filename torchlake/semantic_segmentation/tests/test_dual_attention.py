import pytest
import torch

from torchlake.semantic_segmentation.models.deeplabv2.network import (
    deeplab_v2_style_resnet,
)

from ..models.dual_attention import DANet
from ..models.dual_attention.network import (
    ChannelAttention2d,
    DualAttention2d,
    SpatialAttention2d,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
NUM_CLASS = 21


class TestNetwork:
    def test_spatial_attention_2d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = SpatialAttention2d(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_channel_attention_2d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = ChannelAttention2d()

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_dual_attention_2d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = DualAttention2d(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:

    @pytest.mark.parametrize("is_training", ((True,), (False,)))
    def test_danet_forward_shape(self, is_training: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = DANet(backbone, NUM_CLASS)

        if is_training:
            model.train()
            y, aux = model(x)
            assert y.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )
            assert aux.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )
        else:
            model.eval()
            y = model(x)
            assert y.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )
