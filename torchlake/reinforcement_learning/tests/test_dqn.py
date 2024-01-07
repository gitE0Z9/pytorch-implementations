import torch

from ..models.dqn.model import Dqn
from ..models.dqn.network import QNetwork


def test_forward_shape_network():
    x = torch.rand((2, 1, 128, 128))
    model = QNetwork(5)
    y = model(x)

    assert y.shape == torch.Size((2, 5))


def test_forward_shape_model():
    x = torch.rand((2, 1, 128, 128))
    model = Dqn(5, "cpu")
    y = model(x)

    assert y.shape == torch.Size((2, 5))
