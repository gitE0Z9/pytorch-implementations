import pytest
import torch

from ..models.dilation.model import DilationNet
from ..models.dilation.network import (
    context_module_basic,
    context_module_large,
    dilation_net_style_vgg,
)


BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 320
NUM_CLASS = 21
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8


class TestNetwork:
    def test_dilation_net_style_vgg_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = dilation_net_style_vgg("vgg16", trainable=False)
        model.fix_target_layers(("6_1",))
        y = model(x).pop()

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                model.feature_dim,
                DOWNSCALE_IMAGE_SIZE,
                DOWNSCALE_IMAGE_SIZE,
            )
        )

    @pytest.mark.parametrize(
        "constructor", (context_module_large, context_module_basic)
    )
    def test_context_module_forward_shape(self, constructor):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = constructor(HIDDEN_DIM)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("context_type", ["basic", "large", None])
    def test_dilation_net_forward_shape(self, context_type: str):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = dilation_net_style_vgg("vgg16", trainable=False)
        model = DilationNet(backbone, NUM_CLASS, context_type)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
