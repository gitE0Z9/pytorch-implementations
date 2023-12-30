import pytest
import torch

from ..models import LstmClassifier


@pytest.mark.parametrize(
    "name,label_size,target_shape,num_layers,bidirectional",
    [
        ("single_class", 1, torch.Size((1, 1)), 1, False),
        ("multi_class", 3, torch.Size((1, 256, 3)), 1, False),
        ("bidirectional_single_class", 1, torch.Size((1, 1)), 1, True),
        ("bidirectional_multi_class", 3, torch.Size((1, 256, 3)), 1, True),
        ("two_layer_single_class", 1, torch.Size((1, 1)), 2, False),
        ("two_layer_multi_class", 3, torch.Size((1, 256, 3)), 2, False),
        ("two_layer_bidirectional_single_class", 1, torch.Size((1, 1)), 2, True),
        ("two_layer_bidirectional_multi_class", 3, torch.Size((1, 256, 3)), 2, True),
    ],
)
def test_forward_shape(
    name: str,
    label_size: int,
    target_shape: torch.Size,
    num_layers: int,
    bidirectional: bool,
):
    batch_size, vocab_size = 1, 10
    latent_dim = 8
    x = torch.randint(0, vocab_size, (batch_size, latent_dim))

    model = LstmClassifier(
        vocab_size,
        latent_dim,
        latent_dim,
        label_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    y = model(x)

    assert y.shape == target_shape
