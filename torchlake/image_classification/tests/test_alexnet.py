import torch

from ..models.alexnet.network import LocalResponseNorm

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
OUTPUT_SIZE = 10


class TestNetwork:
    def test_local_response_norm_forward_shape(self):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        m = LocalResponseNorm()

        y = m(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        )
