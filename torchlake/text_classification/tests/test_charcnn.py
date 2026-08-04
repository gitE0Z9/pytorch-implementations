import torch

from torchlake.common.schemas.nlp import NLPContext

from ..models.charcnn import CharCNN

BATCH_SIZE = 10
VOCAB_SIZE = 70
OUTPUT_SIZE = 10
MAX_SEQ_LEN = 27 * 10 + 96
CONTEXT = NLPContext(device="cpu", max_seq_len=MAX_SEQ_LEN)


class TestModel:
    def test_charcnn_forward_shape(self):
        """test output shape"""
        model = CharCNN(VOCAB_SIZE, OUTPUT_SIZE, context=CONTEXT)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
