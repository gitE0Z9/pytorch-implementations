from typing import Sequence
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
        m = fcn_style_vgg("vgg16", trainable=False)
        m.fix_target_layers(("3_1", "4_1", "6_1"))

        features = m(x)

        feature_dims = (m.hidden_dim_8x, m.hidden_dim_16x, m.feature_dim)
        scales = (8, 16, 32)
        for f, d, _ in zip(features, feature_dims, scales):
            assert f.shape[1] == d


class TestModel:
    @pytest.mark.parametrize(
        "output_stride,target_layers",
        (
            (32, ("6_1",)),
            (16, ("4_1", "6_1")),
            (8, ("3_1", "4_1", "6_1")),
        ),
    )
    def test_fcn_forward_shape(self, output_stride: int, target_layers: Sequence[int]):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = fcn_style_vgg("vgg16", trainable=False)
        backbone.fix_target_layers(target_layers)
        model = FCN(backbone, NUM_CLASS, output_stride=output_stride)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
