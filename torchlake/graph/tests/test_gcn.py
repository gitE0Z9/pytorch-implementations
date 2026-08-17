import torch

from torchlake.common.utils.sparse import eye_matrix

from ..models.gcn.model import GCN, GCNResidual
from ..models.gcn.network import GCNLayer, GCNResBlock

NODE_SIZE = 3
INPUT_CHANNEL = 5
OUTPUT_SIZE = 10
HIDDEN_DIM = 8


class TestNetwork:
    def test_gcn_layer_forward_shape(self):
        x = torch.rand((NODE_SIZE, INPUT_CHANNEL))
        a = eye_matrix(NODE_SIZE)
        layer = GCNLayer(INPUT_CHANNEL, OUTPUT_SIZE)

        output = layer(x, a)

        assert output.shape == torch.Size((NODE_SIZE, OUTPUT_SIZE))
        assert not torch.isnan(output).any()

    def test_gcn_resblock_forward_shape(self):
        x = torch.rand((NODE_SIZE, HIDDEN_DIM))
        a = eye_matrix(NODE_SIZE)
        layer = GCNResBlock(HIDDEN_DIM, HIDDEN_DIM)

        output = layer(x, a)

        assert output.shape == torch.Size((NODE_SIZE, HIDDEN_DIM))
        assert not torch.isnan(output).any()


class TestModel:
    def test_gcn_forward_shape(self):
        x = torch.rand((NODE_SIZE, INPUT_CHANNEL))
        a = eye_matrix(NODE_SIZE)
        model = GCN(INPUT_CHANNEL, HIDDEN_DIM, OUTPUT_SIZE)

        output = model(x, a)

        assert output.shape == torch.Size((NODE_SIZE, OUTPUT_SIZE))
        assert not torch.isnan(output).any()

    def test_gcn_residual_forward_shape(self):
        x = torch.rand((NODE_SIZE, INPUT_CHANNEL))
        a = eye_matrix(NODE_SIZE)
        model = GCNResidual(INPUT_CHANNEL, HIDDEN_DIM, OUTPUT_SIZE)

        output = model(x, a)

        assert output.shape == torch.Size((NODE_SIZE, OUTPUT_SIZE))
        assert not torch.isnan(output).any()
