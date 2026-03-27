import torch

from ..models.srcnn.model import SRCNN

BATCH_SIZE = 4
INPUT_CHANNEL = 3
IMAGE_SIZE = 224


class TestModel:
    def test_srcnn_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        model = SRCNN(INPUT_CHANNEL, (9, 1, 5), (64, 32))

        y = model(x)

        OUTPUT_SIZE = IMAGE_SIZE - sum(model.kernels) + len(model.kernels)
        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, OUTPUT_SIZE, OUTPUT_SIZE)
        )
