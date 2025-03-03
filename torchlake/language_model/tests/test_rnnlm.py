import torch
from torchlake.common.schemas.nlp import NlpContext
from torchlake.sequence_data.models.base import RNNGenerator
from torchlake.sequence_data.models.lstm import LSTMDiscriminator

from ..models.rnnlm import RNNLM

BATCH_SIZE = 2
VOCAB_SIZE = 16
MAX_SEQ_LEN = 16
EMBED_SIZE = 8
HIDDEN_DIM = 8
CONTEXT = NlpContext(device="cpu", max_seq_len=MAX_SEQ_LEN)


class TestModel:
    def test_forward_shape(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))

        model = RNNLM(
            backbone=RNNGenerator(
                LSTMDiscriminator(
                    VOCAB_SIZE,
                    EMBED_SIZE,
                    HIDDEN_DIM,
                    VOCAB_SIZE,
                    context=CONTEXT,
                )
            ),
            context=CONTEXT,
        )

        model.train()
        y = model(x)

        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] <= MAX_SEQ_LEN
        assert y.shape[2] == VOCAB_SIZE
