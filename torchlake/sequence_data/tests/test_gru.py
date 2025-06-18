import pytest
import torch
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence

from ..models.base import RNNGenerator
from ..models.gru import GRUDiscriminator
from ..models.gru.network import GRUCell, GRULayer

BATCH_SIZE = 2
VOCAB_SIZE = 10
SEQ_LEN = 16
EMBED_DIM = 16
HIDDEN_DIM = 8
CONTEXT = NlpContext(device="cpu", max_seq_len=SEQ_LEN)


class TestCell:
    def test_forward_shape(self):
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        h = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        model = GRUCell(EMBED_DIM, HIDDEN_DIM)

        h = model(x, h)

        assert h.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM))


class TestLayer:
    def test_forward_shape(self):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
        h = torch.randn(BATCH_SIZE, HIDDEN_DIM)

        model = GRULayer(EMBED_DIM, HIDDEN_DIM)

        h = model(x, h)

        assert h.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestDiscriminator:
    @pytest.mark.parametrize(
        "name,label_size,target_shape,num_layers,bidirectional,sequence_output",
        [
            ("single_class", 1, torch.Size((2, 1)), 1, False, False),
            ("multi_class", 3, torch.Size((2, 3)), 1, False, False),
            ("token_single_class", 1, torch.Size((2, SEQ_LEN, 1)), 1, False, True),
            ("token_multi_class", 3, torch.Size((2, SEQ_LEN, 3)), 1, False, True),
            ("bidirectional_single_class", 1, torch.Size((2, 1)), 1, True, False),
            ("bidirectional_multi_class", 3, torch.Size((2, 3)), 1, True, False),
            (
                "bidirectional_token_single_class",
                1,
                torch.Size((2, SEQ_LEN, 1)),
                1,
                True,
                True,
            ),
            (
                "bidirectional_token_multi_class",
                3,
                torch.Size((2, SEQ_LEN, 3)),
                1,
                True,
                True,
            ),
            ("two_layer_single_class", 1, torch.Size((2, 1)), 2, False, False),
            ("two_layer_multi_class", 3, torch.Size((2, 3)), 2, False, False),
            (
                "two_layer_token_single_class",
                1,
                torch.Size((2, SEQ_LEN, 1)),
                2,
                False,
                True,
            ),
            (
                "two_layer_token_multi_class",
                3,
                torch.Size((2, SEQ_LEN, 3)),
                2,
                False,
                True,
            ),
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
                torch.Size((2, SEQ_LEN, 1)),
                2,
                True,
                True,
            ),
            (
                "two_layer_token_bidirectional_multi_class",
                3,
                torch.Size((2, SEQ_LEN, 3)),
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
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        model = GRUDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            label_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            sequence_output=sequence_output,
        )

        y = model(x)

        assert y.shape == target_shape


class TestGenerator:
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_forward_shape_during_train(
        self,
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        discriminator = GRUDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            VOCAB_SIZE,
            num_layers=num_layers,
            bidirectional=bidirectional,
            context=CONTEXT,
        )

        model = RNNGenerator(discriminator)
        model.train()

        y = model.loss_forward(y)

        assert y.shape[0] == BATCH_SIZE
        assert 0 <= y.shape[1] <= SEQ_LEN
        assert y.shape[2] == VOCAB_SIZE

    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("topk", [1, 3])
    def test_forward_shape_during_predict(
        self,
        num_layers: int,
        bidirectional: bool,
        topk: int,
    ) -> None:
        x = get_input_sequence((BATCH_SIZE, 1), CONTEXT)

        discriminator = GRUDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            VOCAB_SIZE,
            num_layers=num_layers,
            bidirectional=bidirectional,
            context=CONTEXT,
        )

        model = RNNGenerator(discriminator)
        model.eval()

        y = model.predict(x, topk=topk)

        assert y.size(0) == BATCH_SIZE
        assert y.size(1) <= SEQ_LEN + 1
