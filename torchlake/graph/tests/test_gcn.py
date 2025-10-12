import torch

from torchlake.common.utils.sparse import eye_matrix

from ..models.gcn.model import GCN, GCNResidual
from ..models.gcn.network import GCNLayer, GCNResBlock


class TestNetwork:
    def test_gcn_layer_forward_shape(self):
        x = torch.rand((3, 5))
        a = eye_matrix(3)
        layer = GCNLayer(5, 10)

        output = layer(x, a)

        assert output.shape == torch.Size((3, 10))
        assert not torch.isnan(output).any()

    def test_gcn_resblock_forward_shape(self):
        x = torch.rand((3, 10))
        a = eye_matrix(3)
        layer = GCNResBlock(10, 10)

        output = layer(x, a)

        assert output.shape == torch.Size((3, 10))
        assert not torch.isnan(output).any()


class TestModel:
    def test_gcn_forward_shape(self):
        x = torch.rand((3, 5))
        a = eye_matrix(3)
        model = GCN(5, 10, 3)

        output = model(x, a)

        assert output.shape == torch.Size((3, 3))
        assert not torch.isnan(output).any()

    def test_gcn_residual_forward_shape(self):
        x = torch.rand((3, 5))
        a = eye_matrix(3)
        model = GCNResidual(5, 10, 3)

        output = model(x, a)

        assert output.shape == torch.Size((3, 3))
        assert not torch.isnan(output).any()
