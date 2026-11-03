import pytest
import torch

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.sequence import get_input_sequence
from torchlake.sequence_data.models.gru import GRUDiscriminator
from torchlake.sequence_data.models.lstm import LSTMDiscriminator
from torchlake.sequence_data.models.rnn import SimpleRNNDiscriminator

from ..models.rnnlm import RNNLM

BATCH_SIZE = 2
VOCAB_SIZE = 16
MAX_SEQ_LEN = 32
HIDDEN_DIM = 8
CONTEXT = NLPContext(device="cpu", max_seq_len=MAX_SEQ_LEN)


class TestModel:
    @pytest.mark.parametrize(
        "rnn_discriminator_class,input_to_hidden_dim",
        (
            (SimpleRNNDiscriminator, HIDDEN_DIM),
            (LSTMDiscriminator, HIDDEN_DIM * 4),
            (GRUDiscriminator, HIDDEN_DIM * 3),
        ),
    )
    def test_rnnlm_input_embedding(
        self,
        rnn_discriminator_class,
        input_to_hidden_dim: int,
    ):
        model = RNNLM(
            VOCAB_SIZE,
            HIDDEN_DIM,
            rnn_discriminator_class=rnn_discriminator_class,
            context=CONTEXT,
        )

        e = model.input_embedding
        assert e.weight.data.shape == torch.Size((VOCAB_SIZE, input_to_hidden_dim))

    @pytest.mark.parametrize(
        "rnn_discriminator_class",
        (SimpleRNNDiscriminator, LSTMDiscriminator, GRUDiscriminator),
    )
    def test_rnnlm_output_embedding(self, rnn_discriminator_class):
        model = RNNLM(
            VOCAB_SIZE,
            HIDDEN_DIM,
            rnn_discriminator_class=rnn_discriminator_class,
            context=CONTEXT,
        )

        e = model.output_embedding
        assert e.weight.data.shape == torch.Size((VOCAB_SIZE, HIDDEN_DIM))

    @pytest.mark.parametrize(
        "training,x",
        (
            (True, torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN - 1))),
            (False, get_input_sequence((BATCH_SIZE, 1), CONTEXT)),
        ),
    )
    def test_rnnlm_forward_shape(self, training: bool, x: torch.Tensor):
        model = RNNLM(
            VOCAB_SIZE,
            HIDDEN_DIM,
            rnn_discriminator_class=LSTMDiscriminator,
            context=CONTEXT,
        )

        if training:
            model.train()
            y = model(x)
            assert y.shape == torch.Size((BATCH_SIZE, MAX_SEQ_LEN - 1, VOCAB_SIZE))
        else:
            model.eval()
            y = model(x)
            assert y.shape[0] == BATCH_SIZE
            assert y.shape[1] <= MAX_SEQ_LEN
