import torch
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.tree import HuffmanNode

from ..constants.enum import LossType, Word2VecModelType
from ..models.word2vec.loss import HierarchicalSoftmax, NegativeSampling
from ..models.word2vec.model import Word2Vec

BATCH_SIZE = 2
VOCAB_SIZE = 16
CONTEXT_SIZE = 5
EMBED_SIZE = 8
NEIGHBOR_SIZE = CONTEXT_SIZE - 1
SUBSEQ_LEN = 256 - NEIGHBOR_SIZE
NEGATIVE_RATIO = 5
CONTEXT = NlpContext(device="cpu")
TOP_WORD_COUNT = 10
WORD_FREQS = torch.rand((VOCAB_SIZE))
WORD_COUNTS = torch.randint(0, TOP_WORD_COUNT, (VOCAB_SIZE,))


class TestWord2Vec:
    def test_cbow_forward_shape(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_SIZE,
            model_type=Word2VecModelType.CBOW,
            context=CONTEXT,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, 1, SUBSEQ_LEN, VOCAB_SIZE))

    def test_skipgram_forward_shape(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_SIZE,
            Word2VecModelType.SKIP_GRAM,
            context=CONTEXT,
        )
        y = model(x, NEIGHBOR_SIZE)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, VOCAB_SIZE)
        )

    def test_word2vec_subsampling(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        p = torch.randn(VOCAB_SIZE).softmax(0)
        y = Word2Vec.subsampling(x, p, CONTEXT.unk_idx)

        assert y.shape == torch.Size((BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))


class TestNegativeSampling:
    def test_ns_get_distribution_shape(self):
        criterion = NegativeSampling(
            WORD_FREQS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )

        assert criterion.distribution.shape == torch.Size((VOCAB_SIZE,))

    def test_ns_sample_shape(self):
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        criterion = NegativeSampling(
            WORD_FREQS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        y = criterion.sample(y)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, NEGATIVE_RATIO)
        )

    def test_ns_forward(self):
        x = torch.randn(BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, EMBED_SIZE)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))

        criterion = NegativeSampling(
            WORD_FREQS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(x, y)

        assert not torch.isnan(loss)

    def test_ns_backward(self):
        x = torch.randn(BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, EMBED_SIZE)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))

        criterion = NegativeSampling(
            WORD_FREQS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(x, y)
        loss.backward()

        assert not torch.isnan(criterion.fc.grad).any()

    def test_cbow_ns_forward(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_SIZE,
            Word2VecModelType.CBOW,
            loss_type=LossType.NEGATIVE_SAMPLING,
            context=CONTEXT,
        )
        yhat = model(x)

        criterion = NegativeSampling(
            WORD_FREQS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(yhat, y)

        assert not torch.isnan(loss)

    def test_sg_ns_forward(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_SIZE,
            Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.NEGATIVE_SAMPLING,
            context=CONTEXT,
        )
        yhat = model(x, NEIGHBOR_SIZE)

        criterion = NegativeSampling(
            WORD_FREQS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(yhat, y)

        assert not torch.isnan(loss)


class TestHierarchicalSoftmax:
    def test_hs_build_tree(self):
        criterion = HierarchicalSoftmax(
            WORD_COUNTS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        root = criterion.build_tree(WORD_COUNTS)

        assert isinstance(root, HuffmanNode)
        assert root.value > 0
        assert root.left is not None
        assert root.right is not None

    def test_hs_get_paths(self):
        criterion = HierarchicalSoftmax(
            WORD_COUNTS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        root = criterion.build_tree(WORD_COUNTS)
        path_indices, path_codes = criterion.get_paths(root)

        assert len(path_indices) == VOCAB_SIZE
        assert len(path_codes) == VOCAB_SIZE
        for path_idx, path_code in zip(path_indices.values(), path_codes.values()):
            assert path_code.size(0) == path_idx.size(0)
            assert path_code.size(0) <= VOCAB_SIZE
            assert path_idx.size(0) <= VOCAB_SIZE

    def test_hs_forward(self):
        x = torch.randn(BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, EMBED_SIZE)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))

        criterion = HierarchicalSoftmax(
            WORD_COUNTS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(x, y)

        assert not torch.isnan(loss)

    def test_hs_backward(self):
        x = torch.randn(BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN, EMBED_SIZE)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))

        criterion = HierarchicalSoftmax(
            WORD_COUNTS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(x, y)
        loss.backward()

        assert not torch.isnan(criterion.fc.grad).any()

    def test_cbow_hs_forward(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_SIZE,
            Word2VecModelType.CBOW,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
        )
        yhat = model(x)

        criterion = HierarchicalSoftmax(
            WORD_COUNTS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(yhat, y)

        assert not torch.isnan(loss)

    def test_sg_hs_forward(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_SIZE,
            Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
        )
        yhat = model(x, NEIGHBOR_SIZE)

        criterion = HierarchicalSoftmax(
            WORD_COUNTS,
            EMBED_SIZE,
            VOCAB_SIZE,
            context=CONTEXT,
        )
        loss = criterion(yhat, y)

        assert not torch.isnan(loss)
