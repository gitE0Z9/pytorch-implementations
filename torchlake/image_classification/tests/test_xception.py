import torch

from ..models.xception import Xception

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 299
NUM_CLASS = 16


class TestModel:
    def test_xception_forward_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = Xception(output_size=NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS))
