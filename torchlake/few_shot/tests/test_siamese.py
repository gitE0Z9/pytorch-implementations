import torch

from ..models.siamese.model import SiameseNet

INPUT_CHANNEL = 1
IMAGE_SIZE = 105
HIDDEN_DIM = 64
NUM_CLASS = 20
BATCH_SIZE = 32


class TestModel:
    def test_siamese_net_forward_shape(self):
        q = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        s = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = SiameseNet(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        )
        y = model(q, s)

        assert y.shape == torch.Size((BATCH_SIZE, 1))
