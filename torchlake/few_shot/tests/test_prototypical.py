import pytest
import torch

from ..models.prototypical.model import PrototypicalNetwork
from ..models.prototypical.loss import PrototypicalNetworkLoss

QUERY_SIZE = 2
INPUT_CHANNEL = 1
IMAGE_SIZE = 28
HIDDEN_DIM = 64
N_WAY = 5
K_SHOT = 5
NUM_CLASS = 20


class TestModel:
    def test_prototypical_network_feature_extract_shape(self):
        q = torch.randn(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = PrototypicalNetwork(INPUT_CHANNEL, HIDDEN_DIM)
        y = m.feature_extract(q.view(-1, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        assert y.shape == torch.Size((N_WAY * QUERY_SIZE, HIDDEN_DIM))

    def test_prototypical_network_forward_shape(self):
        q = torch.randn(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        support = torch.randn(N_WAY, K_SHOT, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = PrototypicalNetwork(INPUT_CHANNEL, HIDDEN_DIM)

        y = m(q, support)

        assert y.shape == torch.Size((QUERY_SIZE, N_WAY, N_WAY))


class TestLoss:
    def test_prototypical_network_loss_forward(self):
        q = torch.randn(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        support = torch.randn(N_WAY, K_SHOT, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        m = PrototypicalNetwork(INPUT_CHANNEL, HIDDEN_DIM)
        yhat = m(q, support)

        criterion = PrototypicalNetworkLoss()
        loss = criterion(yhat)

        assert not torch.isnan(loss)
