import torch

from ..models.rcnn import RCNN

BATCH_SIZE = 4
VOCAB_SIZE = 100
EMBED_DIM = 8
HIDDEN_DIM = 8
OUTPUT_SIZE = 10
MAX_SEQ_LEN = 256


class TestModel:
    def test_rcnn_forward_shape(self):
        model = RCNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_SIZE)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
