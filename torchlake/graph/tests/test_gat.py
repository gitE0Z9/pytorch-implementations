import pytest
import torch

from ..models.gat.model import GAT
from ..models.gat.network import GATLayer, GATLayerV2, Block

NODE_SIZE = 4
NUM_HEADS = 3
INPUT_DIM = 8
HIDDEN_DIM = 16
OUT_DIM = 4


class TestNetwork:
    @pytest.mark.parametrize(
        "name,predict,output_dim",
        [
            ["hidden-layer", False, NUM_HEADS * HIDDEN_DIM],
            ["output-layer", True, HIDDEN_DIM],
        ],
    )
    def test_gat_layer_forward_shape(self, name: str, predict: bool, output_dim: int):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        layer = GATLayer(INPUT_DIM, HIDDEN_DIM, NUM_HEADS)

        output = layer(x, e, predict=predict)

        assert output.shape == torch.Size((NODE_SIZE, output_dim))

    @pytest.mark.parametrize(
        "name,predict,output_dim",
        [
            ["hidden-layer", False, NUM_HEADS * HIDDEN_DIM],
            ["output-layer", True, HIDDEN_DIM],
        ],
    )
    def test_gat_layer_v2_forward_shape(
        self, name: str, predict: bool, output_dim: int
    ):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        layer = GATLayerV2(INPUT_DIM, HIDDEN_DIM, NUM_HEADS)

        output = layer(x, e, predict=predict)

        assert output.shape == torch.Size((NODE_SIZE, output_dim))

    @pytest.mark.parametrize(
        "name,predict,output_dim",
        [
            ["hidden-layer", False, NUM_HEADS * HIDDEN_DIM],
            ["output-layer", True, HIDDEN_DIM],
        ],
    )
    @pytest.mark.parametrize("version", [1, 2])
    def test_block_forward_shape(
        self,
        name: str,
        predict: bool,
        output_dim: int,
        version: 1 | 2,
    ):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        layer = Block(
            INPUT_DIM,
            HIDDEN_DIM,
            num_heads=NUM_HEADS,
            dropout_prob=0.5,
            version=version,
        )

        output = layer(x, e, predict=predict)

        assert output.shape == torch.Size((NODE_SIZE, output_dim))


class TestModel:
    @pytest.mark.parametrize(
        "name,version",
        [
            ["v1", 1],
            ["v2", 2],
        ],
    )
    @pytest.mark.parametrize("num_block", [0, 1, 2])
    def test_gat_forward_shape(self, name: str, version: int, num_block: int):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        model = GAT(
            INPUT_DIM,
            HIDDEN_DIM,
            OUT_DIM,
            num_heads=NUM_HEADS,
            num_block=num_block,
            version=version,
        )

        output = model(x, e)

        assert output.shape == torch.Size((NODE_SIZE, OUT_DIM))
