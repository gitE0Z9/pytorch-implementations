import torch
from torchlake.common.utils.sparse import eye_matrix

from ..models import Gat
from ..models.attention.model import Gat
from ..models.attention.network import GatLayer


def test_forward_shape_layer():
    x = torch.rand((3, 4))
    a = eye_matrix(3)
    layer = GatLayer(4, 8)

    output = layer(x, a, predict=False)

    assert output.shape == torch.Size((3, 8))


def test_forward_shape_layer_head():
    x = torch.rand((3, 4))
    a = eye_matrix(3)
    layer = GatLayer(4, 8)

    output = layer(x, a, predict=True)

    assert output.shape == torch.Size((3, 8))


def test_forward_shape_model_single_class():
    x = torch.rand((3, 4))
    a = eye_matrix(3)
    model = Gat(4, 8, 1)

    output = model(x, a)

    assert output.shape == torch.Size((3, 1))


def test_forward_shape_model_multi_class():
    x = torch.rand((3, 4))
    a = eye_matrix(3)
    model = Gat(4, 8, 10)

    output = model(x, a)

    assert output.shape == torch.Size((3, 10))
