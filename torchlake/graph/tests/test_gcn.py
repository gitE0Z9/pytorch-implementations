import torch
from torchlake.common.utils.sparse import eye_matrix

from ..models import Gcn
from ..models.gcn.model import GcnResidual
from ..models.gcn.network import GcnLayer, GcnResBlock


def test_forward_shape_layer():
    x = torch.rand((3, 5))
    a = eye_matrix(3)
    layer = GcnLayer(5, 10)

    output = layer(x, a)

    assert output.shape == torch.Size((3, 10))


def test_forward_nan_layer():
    x = torch.rand((3, 5))
    a = eye_matrix(3)
    layer = GcnLayer(5, 10)

    output = layer(x, a)

    assert not torch.isnan(output).any()


def test_forward_shape_block():
    x = torch.rand((3, 10))
    a = eye_matrix(3)
    layer = GcnResBlock(10, 10)

    output = layer(x, a)

    assert output.shape == torch.Size((3, 10))


def test_forward_nan_block():
    x = torch.rand((3, 10))
    a = eye_matrix(3)
    layer = GcnResBlock(10, 10)

    output = layer(x, a)

    assert not torch.isnan(output).any()


def test_forward_shape_model():
    x = torch.rand((3, 5))
    a = eye_matrix(3)
    model = Gcn(5, 10, 3)

    output = model(x, a)

    assert output.shape == torch.Size((3, 3))


def test_forward_nan_model():
    x = torch.rand((3, 5))
    a = eye_matrix(3)
    model = Gcn(5, 10, 3)

    output = model(x, a)

    assert not torch.isnan(output).any()


def test_forward_shape_res_model():
    x = torch.rand((3, 5))
    a = eye_matrix(3)
    model = GcnResidual(5, 10, 3)

    output = model(x, a)

    assert output.shape == torch.Size((3, 3))


def test_forward_nan_res_model():
    x = torch.rand((3, 5))
    a = eye_matrix(3)
    model = GcnResidual(5, 10, 3)

    output = model(x, a)

    assert not torch.isnan(output).any()
