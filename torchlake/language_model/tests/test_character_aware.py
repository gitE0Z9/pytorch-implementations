import random

import torch

from torchlake.common.schemas.nlp import NLPContext

from ..models.character_aware.model import CharaceterAwareLM
from ..models.character_aware.network import CharacterCNNEmbedding

CHAR_SIZE = 70
BATCH_SIZE = 2
VOCAB_SIZE = 16
EMBED_SIZE = 8
# simulate real situation
MIN_CHAR_IN_A_WORD = 1
MAX_CHAR_IN_A_WORD = 90
SEQ_LEN = 256
MIN_SEQ_LEN = 5

MAX_CHAR_SIZE = 65
HIDDEN_DIMS = [8, 16, 24]
KERNELS = [1, 2, 3]
CONTEXT = NLPContext(device="cpu")


class TestNetwork:
    def setUp(self) -> tuple[list[int], list[torch.Tensor], list]:
        seq_lens = [random.randint(MIN_SEQ_LEN, SEQ_LEN) for _ in range(BATCH_SIZE)]
        word_spans = [
            torch.randint(MIN_CHAR_IN_A_WORD, MAX_CHAR_IN_A_WORD, (seq_len,))
            for seq_len in seq_lens
        ]
        ngrams = [
            torch.randint(0, CHAR_SIZE, (word_span.sum().item(),))
            for word_span in word_spans
        ]

        return seq_lens, word_spans, ngrams

    def test_character_cnn_embedding_forward_shape(self):
        seq_lens, word_spans, ngrams = self.setUp()

        model = CharacterCNNEmbedding(
            CHAR_SIZE,
            EMBED_SIZE,
            hidden_dims=HIDDEN_DIMS,
            kernels=KERNELS,
            max_char_size=MAX_CHAR_SIZE,
            context=CONTEXT,
        )

        y = model(ngrams, word_spans)

        assert y.shape == torch.Size((sum(seq_lens), sum(HIDDEN_DIMS)))


class TestModel:
    def setUp(self) -> tuple[list[int], list[torch.Tensor], list]:
        seq_lens = [random.randint(MIN_SEQ_LEN, SEQ_LEN) for _ in range(BATCH_SIZE)]
        word_spans = [
            torch.randint(MIN_CHAR_IN_A_WORD, MAX_CHAR_IN_A_WORD, (seq_len,))
            for seq_len in seq_lens
        ]
        ngrams = [
            torch.randint(0, CHAR_SIZE, (word_span.sum().item(),))
            for word_span in word_spans
        ]

        return seq_lens, word_spans, ngrams

    def test_character_aware_lm_get_word_vectors_shape(self):
        _, word_spans, ngrams = self.setUp()

        model = CharaceterAwareLM(
            CHAR_SIZE,
            VOCAB_SIZE,
            EMBED_SIZE,
            hidden_dims=HIDDEN_DIMS,
            kernels=KERNELS,
            recurrent_hidden_dim=32,
            max_char_size=MAX_CHAR_SIZE,
            context=CONTEXT,
        )

        y = model.get_word_vectors(ngrams, word_spans)

        assert len(y) == BATCH_SIZE
        for it in y:
            s, h = it.shape
            assert s <= SEQ_LEN
            assert h == sum(HIDDEN_DIMS)

    def test_character_aware_lm_forward_shape(self):
        _, word_spans, ngrams = self.setUp()

        model = CharaceterAwareLM(
            CHAR_SIZE,
            VOCAB_SIZE,
            EMBED_SIZE,
            hidden_dims=HIDDEN_DIMS,
            kernels=KERNELS,
            recurrent_hidden_dim=32,
            max_char_size=MAX_CHAR_SIZE,
            context=CONTEXT,
        )

        y = model(
            [it[: -word_spans[i][-1].item()] for i, it in enumerate(ngrams)],
            [it[:-1] for it in word_spans],
        )

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN - 1, VOCAB_SIZE))
