import torch

from ..models.siamese.model import SiameseNet
from ..models.siamese.loss import SiameseNetLoss

QUERY_SIZE = 2
INPUT_CHANNEL = 1
IMAGE_SIZE = 105
HIDDEN_DIM = 64
N_WAY = 5
K_SHOT = 1
NUM_CLASS = 20


class TestModel:
    def test_siamese_net_forward_shape(self):
        q = torch.randn(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        s = torch.randn(N_WAY, K_SHOT, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = SiameseNet(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        )
        y = model(q, s)

        assert y.shape == torch.Size((QUERY_SIZE, N_WAY, N_WAY))


class TestLoss:
    def test_siamese_net_loss_forward(self):
        q = torch.randn(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        s = torch.randn(N_WAY, K_SHOT, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = SiameseNet(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        )
        yhat = model(q, s)

        criterion = SiameseNetLoss()
        loss = criterion(yhat)

        assert not torch.isnan(loss)
