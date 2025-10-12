import pytest
import torch

from ..models.attention.model import GAT
from ..models.attention.network import GATLayer, GATLayerV2

NODE_SIZE = 4
NUM_HEADS = 3
INPUT_DIM = 8
LATENT_DIM = 16
OUT_DIM = 4


class TestNetwork:
    @pytest.mark.parametrize(
        "name,predict,output_dim",
        [
            ["hidden-layer", False, NUM_HEADS * LATENT_DIM],
            ["output-layer", True, LATENT_DIM],
        ],
    )
    def test_gat_layer_forward_shape(self, name: str, predict: bool, output_dim: int):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        layer = GATLayer(INPUT_DIM, LATENT_DIM, NUM_HEADS)

        output = layer(x, e, predict=predict)

        assert output.shape == torch.Size((NODE_SIZE, output_dim))

    @pytest.mark.parametrize(
        "name,predict,output_dim",
        [
            ["hidden-layer", False, NUM_HEADS * LATENT_DIM],
            ["output-layer", True, LATENT_DIM],
        ],
    )
    def test_gat_layer_v2_forward_shape(
        self, name: str, predict: bool, output_dim: int
    ):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        layer = GATLayerV2(INPUT_DIM, LATENT_DIM, NUM_HEADS)

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
    def test_gat_forward_shape(self, name: str, version: int):
        x = torch.rand((NODE_SIZE, INPUT_DIM))
        e = torch.LongTensor([[0, 0], [1, 1], [2, 3], [3, 0]])
        model = GAT(
            INPUT_DIM, LATENT_DIM, OUT_DIM, num_heads=NUM_HEADS, version=version
        )

        output = model(x, e)

        assert output.shape == torch.Size((NODE_SIZE, OUT_DIM))
