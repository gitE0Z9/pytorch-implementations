import pytest
import torch
from torch import nn

from ...common.schemas.nlp import NlpContext
from ..constants.enum import LossType, NgramCombinationMethod, Word2VecModelType
from ..models.subword.model import SubwordLM
from ..models.subword.network import SubwordEmbedding
from ..models.word2vec.loss import HierarchicalSoftmax, NegativeSampling

BATCH_SIZE = 16
BUCKET_SIZE = 100
EMBED_DIM = 300
NEIGHBOR_SIZE = 4
VOCAB_SIZE = 100
SUBSEQ_LEN = 256
CONTEXT = NlpContext(max_seq_len=SUBSEQ_LEN, device="cpu")
WORD_COUNTS = torch.randint(0, 5, (VOCAB_SIZE,))


class TestSubwordEmbedding:
    @pytest.mark.parametrize(
        "combination",
        [
            NgramCombinationMethod.NGRAM_ONLY,
            NgramCombinationMethod.WORD_AND_NGRAM,
        ],
    )
    def test_forward_shape(self, combination: int):
        ngram = [
            torch.randint(0, VOCAB_SIZE, (SUBSEQ_LEN * 2,)) for _ in range(BATCH_SIZE)
        ]
        word = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SUBSEQ_LEN))
        word_span = [torch.ones_like(w) * 2 for w in word]

        model = SubwordEmbedding(
            BUCKET_SIZE,
            EMBED_DIM,
            combination=combination,
            context=CONTEXT,
        )

        y = model.forward(ngram, word, word_span)

        assert y.shape == torch.Size((BATCH_SIZE, SUBSEQ_LEN, EMBED_DIM))


class TestSubWordLM:
    def setUp(self):
        # gram
        self.gram_ngram = [
            torch.randint(0, VOCAB_SIZE, (SUBSEQ_LEN * 2,))
            for _ in range(BATCH_SIZE * 1)
        ]
        self.gram_word = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        self.gram_word_span = [
            torch.ones_like(w) * 2 for w in self.gram_word.view(-1, SUBSEQ_LEN)
        ]

        # context
        self.context_ngram = [
            torch.randint(0, VOCAB_SIZE, (SUBSEQ_LEN * 2,))
            for _ in range(BATCH_SIZE * NEIGHBOR_SIZE)
        ]
        self.context_word = torch.randint(
            0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN)
        )
        self.context_word_span = [
            torch.ones_like(w) * 2 for w in self.context_word.view(-1, SUBSEQ_LEN)
        ]

    def test_cb_ce_shape(self):
        self.setUp()

        model = SubwordLM(
            BUCKET_SIZE,
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.CBOW,
            loss_type=LossType.CROSS_ENTROPY,
            context=CONTEXT,
        )

        gram_prob = model.forward(
            self.context_ngram,
            self.context_word.view(-1, SUBSEQ_LEN),
            self.context_word_span,
            BATCH_SIZE,
            1,
        )

        criterion = nn.CrossEntropyLoss()
        loss = criterion.forward(
            gram_prob.permute(0, 3, 1, 2).repeat(1, 1, NEIGHBOR_SIZE, 1),
            self.context_word,
        )
        loss.backward()

        assert gram_prob.shape == torch.Size((BATCH_SIZE, 1, SUBSEQ_LEN, VOCAB_SIZE))
        assert not torch.isnan(loss)

    def test_sg_ce_shape(self):
        self.setUp()

        model = SubwordLM(
            BUCKET_SIZE,
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.CROSS_ENTROPY,
            context=CONTEXT,
        )

        context_prob = model.forward(
            self.gram_ngram,
            self.gram_word.view(-1, SUBSEQ_LEN),
            self.gram_word_span,
            BATCH_SIZE,
            NEIGHBOR_SIZE,
        )

        criterion = nn.CrossEntropyLoss()
        loss = criterion.forward(
            context_prob.permute(0, 3, 1, 2),
            self.context_word,
        )
        loss.backward()

        assert context_prob.shape == torch.Size(
            (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, VOCAB_SIZE)
        )
        assert not torch.isnan(loss)

    def test_cb_ns_shape(self):
        self.setUp()

        model = SubwordLM(
            BUCKET_SIZE,
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.CBOW,
            loss_type=LossType.NEGATIVE_SAMPLING,
            context=CONTEXT,
        )

        gram_prob = model.forward(
            self.context_ngram,
            self.context_word.view(-1, SUBSEQ_LEN),
            self.context_word_span,
            BATCH_SIZE,
            1,
        )

        criterion = NegativeSampling(
            WORD_COUNTS,
            EMBED_DIM,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion.forward(gram_prob, self.context_word)
        loss.backward()

        assert gram_prob.shape == torch.Size((BATCH_SIZE, 1, SUBSEQ_LEN, EMBED_DIM))
        assert not torch.isnan(loss)

    def test_sg_ns_shape(self):
        self.setUp()

        model = SubwordLM(
            BUCKET_SIZE,
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.NEGATIVE_SAMPLING,
            context=CONTEXT,
        )

        context_prob = model.forward(
            self.gram_ngram,
            self.gram_word.view(-1, SUBSEQ_LEN),
            self.gram_word_span,
            BATCH_SIZE,
            NEIGHBOR_SIZE,
        )

        criterion = NegativeSampling(
            WORD_COUNTS,
            EMBED_DIM,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion.forward(context_prob, self.gram_word)
        loss.backward()

        assert context_prob.shape == torch.Size(
            (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, EMBED_DIM)
        )
        assert not torch.isnan(loss)

    def test_cb_hs_shape(self):
        self.setUp()

        model = SubwordLM(
            BUCKET_SIZE,
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.CBOW,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
        )

        gram_prob = model.forward(
            self.context_ngram,
            self.context_word.view(-1, SUBSEQ_LEN),
            self.context_word_span,
            BATCH_SIZE,
            1,
        )

        criterion = HierarchicalSoftmax(WORD_COUNTS, EMBED_DIM, VOCAB_SIZE, CONTEXT)
        loss = criterion.forward(gram_prob, self.gram_word)
        loss.backward()

        assert gram_prob.shape == torch.Size((BATCH_SIZE, 1, SUBSEQ_LEN, EMBED_DIM))
        assert not torch.isnan(loss)

    def test_sg_hs_shape(self):
        self.setUp()

        model = SubwordLM(
            BUCKET_SIZE,
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
        )

        context_prob = model.forward(
            self.gram_ngram,
            self.gram_word.view(-1, SUBSEQ_LEN),
            self.gram_word_span,
            BATCH_SIZE,
            NEIGHBOR_SIZE,
        )

        criterion = HierarchicalSoftmax(WORD_COUNTS, EMBED_DIM, VOCAB_SIZE, CONTEXT)
        loss = criterion.forward(context_prob, self.gram_word)
        loss.backward()

        assert context_prob.shape == torch.Size(
            (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, EMBED_DIM)
        )
        assert not torch.isnan(loss)
