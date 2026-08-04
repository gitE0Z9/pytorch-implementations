import torch

from ..models.fasttext import FastText

BATCH_SIZE = 16
BUCKET_SIZE = 100
EMBED_DIM = 300
OUTPUT_SIZE = 10
VOCAB_SIZE = 100
SUBSEQ_LEN = 256


class TestModel:
    def setUp(self) -> None:
        self.ngram = [
            torch.randint(0, VOCAB_SIZE, (SUBSEQ_LEN * 2,)) for _ in range(BATCH_SIZE)
        ]
        self.word = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUBSEQ_LEN))
        self.word_span = [torch.ones_like(w) * 2 for w in self.word]

    def test_get_sentence_vector_shape(self):
        """test output shape"""
        self.setUp()

        model = FastText(BUCKET_SIZE, EMBED_DIM, OUTPUT_SIZE)
        output = model.get_sentence_vector(self.ngram, self.word, self.word_span)

        assert output.shape == torch.Size((BATCH_SIZE, EMBED_DIM))

    def test_output_shape(self):
        """test output shape"""
        self.setUp()

        model = FastText(BUCKET_SIZE, EMBED_DIM, OUTPUT_SIZE)

        output = model.forward(self.ngram, self.word, self.word_span)

        assert output.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
