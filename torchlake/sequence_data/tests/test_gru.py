import pytest
import torch

from ..models.gru.network import GruCell, GruLayer
from ..models import GruClassifier


def test_forward_shape_cell():
    batch_size = 2
    embed_dim = 16
    latent_dim = 8
    x = torch.randn(batch_size, embed_dim)
    h = torch.randn(batch_size, latent_dim)

    model = GruCell(embed_dim, latent_dim)

    h = model(x, h)

    assert h.shape == torch.Size((batch_size, latent_dim))


@pytest.mark.parametrize(
    "name,label_size,target_shape,num_layers,bidirectional,is_token",
    [
        ("single_class", 1, torch.Size((2, 1)), 1, False, False),
        ("multi_class", 3, torch.Size((2, 3)), 1, False, False),
        ("token_single_class", 1, torch.Size((2, 256, 1)), 1, False, True),
        ("token_multi_class", 3, torch.Size((2, 256, 3)), 1, False, True),
        ("bidirectional_single_class", 1, torch.Size((2, 1)), 1, True, False),
        ("bidirectional_multi_class", 3, torch.Size((2, 3)), 1, True, False),
        ("bidirectional_token_single_class", 1, torch.Size((2, 256, 1)), 1, True, True),
        ("bidirectional_token_multi_class", 3, torch.Size((2, 256, 3)), 1, True, True),
        ("two_layer_single_class", 1, torch.Size((2, 1)), 2, False, False),
        ("two_layer_multi_class", 3, torch.Size((2, 3)), 2, False, False),
        ("two_layer_token_single_class", 1, torch.Size((2, 256, 1)), 2, False, True),
        ("two_layer_token_multi_class", 3, torch.Size((2, 256, 3)), 2, False, True),
        ("two_layer_bidirectional_single_class", 1, torch.Size((2, 1)), 2, True, False),
        ("two_layer_bidirectional_multi_class", 3, torch.Size((2, 3)), 2, True, False),
        (
            "two_layer_token_bidirectional_single_class",
            1,
            torch.Size((2, 256, 1)),
            2,
            True,
            True,
        ),
        (
            "two_layer_token_bidirectional_multi_class",
            3,
            torch.Size((2, 256, 3)),
            2,
            True,
            True,
        ),
    ],
)
def test_forward_shape(
    name: str,
    label_size: int,
    target_shape: torch.Size,
    num_layers: int,
    bidirectional: bool,
    is_token: bool,
) -> None:
    batch_size, vocab_size = 2, 10
    latent_dim = 8
    x = torch.randint(0, vocab_size, (batch_size, latent_dim))

    model = GruClassifier(
        vocab_size,
        latent_dim,
        latent_dim,
        label_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        is_token=is_token,
    )

    y = model(x)

    assert y.shape == target_shape
