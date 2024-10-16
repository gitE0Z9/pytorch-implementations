import pytest
import torch

from ..models.rnn.network import RNNCell, RNNLayer
from ..models.rnn import RNNDiscriminator

BATCH_SIZE = 2
SEQ_LEN = 16
EMBED_DIM = 16
HIDDEN_DIM = 8


class TestCell:
    def test_forward_shape(self):
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        h = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        model = RNNCell(EMBED_DIM, HIDDEN_DIM)

        h = model(x, h)

        assert h.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM))


class TestLayer:
    def test_forward_shape(self):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
        h = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        model = RNNLayer(EMBED_DIM, HIDDEN_DIM)

        h = model(x, h)

        assert h.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestDiscriminator:
    @pytest.mark.parametrize(
        "name,label_size,target_shape,num_layers,bidirectional,sequence_output",
        [
            ("single_class", 1, torch.Size((2, 1)), 1, False, False),
            ("multi_class", 3, torch.Size((2, 3)), 1, False, False),
            ("token_single_class", 1, torch.Size((2, 256, 1)), 1, False, True),
            ("token_multi_class", 3, torch.Size((2, 256, 3)), 1, False, True),
            ("bidirectional_single_class", 1, torch.Size((2, 1)), 1, True, False),
            ("bidirectional_multi_class", 3, torch.Size((2, 3)), 1, True, False),
            (
                "bidirectional_token_single_class",
                1,
                torch.Size((2, 256, 1)),
                1,
                True,
                True,
            ),
            (
                "bidirectional_token_multi_class",
                3,
                torch.Size((2, 256, 3)),
                1,
                True,
                True,
            ),
            ("two_layer_single_class", 1, torch.Size((2, 1)), 2, False, False),
            ("two_layer_multi_class", 3, torch.Size((2, 3)), 2, False, False),
            (
                "two_layer_token_single_class",
                1,
                torch.Size((2, 256, 1)),
                2,
                False,
                True,
            ),
            ("two_layer_token_multi_class", 3, torch.Size((2, 256, 3)), 2, False, True),
            (
                "two_layer_bidirectional_single_class",
                1,
                torch.Size((2, 1)),
                2,
                True,
                False,
            ),
            (
                "two_layer_bidirectional_multi_class",
                3,
                torch.Size((2, 3)),
                2,
                True,
                False,
            ),
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
        self,
        name: str,
        label_size: int,
        target_shape: torch.Size,
        num_layers: int,
        bidirectional: bool,
        sequence_output: bool,
    ) -> None:
        batch_size, vocab_size = 2, 10
        latent_dim = 8
        x = torch.randint(0, vocab_size, (batch_size, latent_dim))

        model = RNNDiscriminator(
            vocab_size,
            latent_dim,
            latent_dim,
            label_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            sequence_output=sequence_output,
        )

        y = model(x)

        assert y.shape == target_shape
