import pytest
import torch

from ..models.fcn.model import FCN
from ..models.fcn.network import fcn_style_vgg

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
NUM_CLASS = 21


class TestNetwork:
    def test_fcn_style_vgg_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        m = fcn_style_vgg(
            "vgg16",
            trainable=False,
            num_skip_connection=2,
        )

        features = m(x)

        feature_dims = (m.hidden_dim_8x, m.hidden_dim_16x, m.feature_dim)
        scales = (8, 16, 32)
        for f, d, _ in zip(features, feature_dims, scales):
            assert f.shape[1] == d


class TestModel:
    @pytest.mark.parametrize("num_skip_connection", (0, 1, 2))
    def test_fcn_forward_shape(self, num_skip_connection: int):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = fcn_style_vgg(
            "vgg16",
            trainable=False,
            num_skip_connection=num_skip_connection,
        )
        model = FCN(
            backbone,
            NUM_CLASS,
            num_skip_connection=num_skip_connection,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
