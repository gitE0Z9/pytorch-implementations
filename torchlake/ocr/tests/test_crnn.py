import torch

from torchlake.common.schemas.nlp import NLPContext

from ..models.crnn.model import CRNN

BATCH_SIZE = 1
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
VOCAB_SIZE = 10
HEIGHT = 32
WIDTH = 1698
CONTEXT = NLPContext(padding_idx=None)


class TestModel:
    def test_crnn_forward_shape(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, HEIGHT, WIDTH)
        model = CRNN(INPUT_CHANNEL, HIDDEN_DIM, VOCAB_SIZE, context=CONTEXT)
        output = model(x)

        assert output.shape == torch.Size((WIDTH // 4 - 1, 1, VOCAB_SIZE))
