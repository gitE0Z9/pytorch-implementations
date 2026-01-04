import torch

from ..models.parsenet.network import parsenet_style_vgg, GlobalContextModule
from ..models.parsenet.model import ParseNet

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
NUM_CLASS = 21


class TestNetwork:
    def test_parsenet_style_vgg_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = parsenet_style_vgg("vgg16", trainable=False)
        model.fix_target_layers(("6_1"))

        y = model(x).pop()

        assert y.shape[:2] == torch.Size((BATCH_SIZE, model.feature_dim))

    def test_global_context_module_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = GlobalContextModule(INPUT_CHANNEL, INPUT_CHANNEL)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )


class TestModel:
    def test_parsenet_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = parsenet_style_vgg("vgg16", trainable=False)
        model = ParseNet(backbone, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
