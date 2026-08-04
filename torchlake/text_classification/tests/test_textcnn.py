import torch

from ..models.textcnn import TextCNN

BATCH_SIZE = 4
VOCAB_SIZE = 26
EMBED_DIM = 8
OUTPUT_SIZE = 10
MAX_SEQ_LEN = 256


class TestModel:
    def test_textcnn_forward_shape(self):
        model = TextCNN(VOCAB_SIZE, EMBED_DIM, output_size=OUTPUT_SIZE)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
