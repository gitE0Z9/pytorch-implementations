import pytest
import torch

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import build_huffman_tree
from torchlake.language_model.constants.enum import LossType

from ..models.fasttext import FastText

BATCH_SIZE = 16
BUCKET_SIZE = 100
EMBED_DIM = 300
OUTPUT_SIZE = 10
VOCAB_SIZE = 100
SUBSEQ_LEN = 256
TOP_WORD_COUNT = 10
WORD_COUNTS = torch.randint(0, TOP_WORD_COUNT, (OUTPUT_SIZE,))
CONTEXT = NLPContext(device="cpu")


class TestModel:
    def setUp(self) -> None:
        self.ngram = [
            torch.randint(0, VOCAB_SIZE, (SUBSEQ_LEN * 2,)) for _ in range(BATCH_SIZE)
        ]
        self.word = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUBSEQ_LEN))
        self.word_span = [torch.ones_like(w) * 2 for w in self.word]

    def test_fasttext_get_sentence_vector_shape(self):
        self.setUp()

        model = FastText(BUCKET_SIZE, EMBED_DIM, OUTPUT_SIZE)
        output = model.get_sentence_vector(self.ngram, self.word, self.word_span)

        assert output.shape == torch.Size((BATCH_SIZE, EMBED_DIM))

    def test_fasttext_forward_shape(self):
        self.setUp()

        model = FastText(BUCKET_SIZE, EMBED_DIM, OUTPUT_SIZE)

        output = model(self.ngram, self.word, self.word_span)

        assert output.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize(
        "is_training,expected_shape",
        ((True, (BATCH_SIZE, EMBED_DIM)), (False, (BATCH_SIZE,))),
    )
    def test_fasttext_hs_forward_shape(self, is_training: bool, expected_shape: tuple):
        self.setUp()

        root = build_huffman_tree(WORD_COUNTS)
        model = FastText(
            BUCKET_SIZE,
            EMBED_DIM,
            OUTPUT_SIZE,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
            root=root,
        )
        model.train(is_training)

        output = model(self.ngram, self.word, self.word_span)

        assert output.shape == torch.Size(expected_shape)
