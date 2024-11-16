import torch

from ..models.transformer.model import TransformerDecoder, TransformerEncoder

BATCH_SIZE = 4
SEQ_LEN = 16
INPUT_VOCAB_SIZE = 10
OUTPUT_VOCAB_SIZE = 15
HIDDEN_DIM = 16


class TestEncoder:
    def test_forward_shape(self):
        x = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = TransformerEncoder(INPUT_VOCAB_SIZE, hidden_dim=HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestDecoder:
    def test_forward_shape(self):
        x = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = TransformerEncoder(INPUT_VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
        model2 = TransformerDecoder(
            OUTPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, hidden_dim=HIDDEN_DIM
        )

        encoded = model(x)
        y = model2(x, encoded)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, OUTPUT_VOCAB_SIZE))
