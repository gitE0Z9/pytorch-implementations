import torch

from ..models.highway.model import HighwayNetwork
from ..models.highway.network import HighwayLayer


def test_layer_forward_shape():
    x = torch.randn(2, 64, 13, 13)
    layer = HighwayLayer(64, 128, 3)
    y = layer(x)

    assert y.shape == torch.Size((2, 128, 13, 13))


def test_network_forward_shape():
    x = torch.randn(2, 3, 224, 224)

    layer = HighwayNetwork(
        [
            ["c", 3, 32, 3],
            ["p", 0, 0, 0],
            ["c", 32, 32, 3],
            ["c", 32, 32, 3],
            ["c", 32, 64, 3],
            ["p", 0, 0, 0],
            ["c", 64, 64, 3],
            ["c", 64, 64, 3],
            ["c", 64, 128, 3],
            ["p", 0, 0, 0],
            ["c", 128, 128, 3],
            ["c", 128, 128, 3],
            ["c", 128, 256, 3],
            ["p", 0, 0, 0],
            ["c", 256, 256, 3],
            ["c", 256, 256, 3],
            ["c", 256, 512, 3],
        ],
    )

    y = layer(x)

    assert y.shape == torch.Size((2, 1))
