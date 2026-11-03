import pytest
import torch
from torch import nn

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import build_huffman_tree

from ..constants.enum import LossType, Word2VecModelType
from ..models.word2vec.loss import HierarchicalSoftmaxLoss, NegativeSampling
from ..models.word2vec.network import HierarchicalSoftmax
from ..models.word2vec.model import Word2Vec
from ..models.word2vec.utils import subsampling

BATCH_SIZE = 2
VOCAB_SIZE = 16
CONTEXT_SIZE = 5
EMBED_DIM = 8
NEIGHBOR_SIZE = CONTEXT_SIZE - 1
SUBSEQ_LEN = 256 - NEIGHBOR_SIZE
NEGATIVE_RATIO = 5
CONTEXT = NLPContext(device="cpu")
TOP_WORD_COUNT = 10
WORD_FREQS = torch.rand((VOCAB_SIZE))
WORD_COUNTS = torch.randint(0, TOP_WORD_COUNT, (VOCAB_SIZE,))


class TestNetwork:
    # def test_hierarchical_softmax_get_logit_shape(self):
    #     x = torch.rand(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)

    #     root = build_huffman_tree(WORD_COUNTS)
    #     model = HierarchicalSoftmax(root, VOCAB_SIZE, EMBED_DIM, CONTEXT)

    #     y = model.get_logit(x)

    #     assert y.shape == torch.Size(
    #         (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, VOCAB_SIZE)
    #     )

    def test_hierarchical_softmax_forward_shape(self):
        x = torch.rand(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)

        root = build_huffman_tree(WORD_COUNTS)
        model = HierarchicalSoftmax(root, VOCAB_SIZE, EMBED_DIM, CONTEXT)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))


class TestModel:
    @pytest.mark.parametrize(
        "loss_type,is_training,expected_shape",
        (
            (LossType.CROSS_ENTROPY, True, (BATCH_SIZE * SUBSEQ_LEN, 1, VOCAB_SIZE)),
            (LossType.CROSS_ENTROPY, False, (BATCH_SIZE * SUBSEQ_LEN, 1, VOCAB_SIZE)),
            (LossType.NEGATIVE_SAMPLING, True, (BATCH_SIZE * SUBSEQ_LEN, 1, EMBED_DIM)),
            (
                LossType.NEGATIVE_SAMPLING,
                False,
                (BATCH_SIZE * SUBSEQ_LEN, 1, VOCAB_SIZE),
            ),
        ),
    )
    def test_cbow_forward_shape(
        self,
        loss_type,
        is_training: bool,
        expected_shape: tuple[int],
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.CBOW,
            loss_type=loss_type,
            context=CONTEXT,
        )
        model.train(is_training)

        y = model(x)

        assert y.shape == torch.Size(expected_shape)

    @pytest.mark.parametrize(
        "loss_type,is_training,expected_shape",
        (
            (
                LossType.CROSS_ENTROPY,
                True,
                (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, VOCAB_SIZE),
            ),
            (
                LossType.CROSS_ENTROPY,
                False,
                (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, VOCAB_SIZE),
            ),
            (
                LossType.NEGATIVE_SAMPLING,
                True,
                (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM),
            ),
            (
                LossType.NEGATIVE_SAMPLING,
                False,
                (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, VOCAB_SIZE),
            ),
        ),
    )
    def test_skipgram_forward_shape(
        self,
        loss_type,
        is_training: bool,
        expected_shape: tuple[int],
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.SKIP_GRAM,
            loss_type=loss_type,
            context=CONTEXT,
        )
        model.train(is_training)

        y = model(x, NEIGHBOR_SIZE)

        assert y.shape == torch.Size(expected_shape)


class TestLoss:
    def test_ns_get_distribution_shape(self):
        criterion = NegativeSampling(WORD_FREQS, context=CONTEXT)

        assert criterion.distribution.shape == torch.Size((VOCAB_SIZE,))

    @pytest.mark.parametrize("replacement", (True, False))
    def test_ns_sample_shape(self, replacement: bool):
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        criterion = NegativeSampling(
            WORD_FREQS,
            replacement=replacement,
            context=CONTEXT,
        )
        y = criterion.sample(y)

        assert y.shape == torch.Size(
            (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, NEGATIVE_RATIO)
        )

    @pytest.mark.parametrize("replacement", (True, False))
    @pytest.mark.parametrize("reduction", ("sum", "mean"))
    def test_ns_forward(self, replacement: bool, reduction: str):
        x = torch.randn(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        e = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        criterion = NegativeSampling(
            WORD_FREQS,
            replacement=replacement,
            reduction=reduction,
            context=CONTEXT,
        )
        loss = criterion(x, y, output_embedding=e)

        assert not torch.isnan(loss)

    @pytest.mark.parametrize("replacement", (True, False))
    @pytest.mark.parametrize("reduction", ("sum", "mean"))
    def test_ns_backward(self, replacement: bool, reduction: str):
        x = torch.randn(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        e = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        criterion = NegativeSampling(
            WORD_FREQS,
            replacement=replacement,
            reduction=reduction,
            context=CONTEXT,
        )
        loss = criterion(x, y, output_embedding=e)
        loss.backward()

        assert not torch.isnan(e.weight.grad).any()

    @pytest.mark.parametrize("replacement", (True, False))
    @pytest.mark.parametrize("reduction", ("sum", "mean"))
    def test_ns_cbow_forward(self, replacement: bool, reduction: str):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.CBOW,
            loss_type=LossType.NEGATIVE_SAMPLING,
            context=CONTEXT,
        )
        yhat = model(x)

        criterion = NegativeSampling(
            WORD_FREQS,
            replacement=replacement,
            reduction=reduction,
            context=CONTEXT,
        )
        loss = criterion(yhat, y, output_embedding=model.output_embedding)

        assert not torch.isnan(loss)

    @pytest.mark.parametrize("replacement", (True, False))
    @pytest.mark.parametrize("reduction", ("sum", "mean"))
    def test_ns_sg_forward(self, replacement: bool, reduction: str):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.NEGATIVE_SAMPLING,
            context=CONTEXT,
        )
        yhat = model(x, NEIGHBOR_SIZE)

        criterion = NegativeSampling(
            WORD_FREQS,
            replacement=replacement,
            reduction=reduction,
            context=CONTEXT,
        )
        loss = criterion(yhat, y, output_embedding=model.output_embedding)

        assert not torch.isnan(loss)

    def test_hs_get_paths(self):
        root = build_huffman_tree(WORD_COUNTS)
        criterion = HierarchicalSoftmaxLoss(root, VOCAB_SIZE, context=CONTEXT)

        assert len(criterion.path_indices) == VOCAB_SIZE
        assert len(criterion.path_codes) == VOCAB_SIZE
        for path_idx, path_code in zip(
            criterion.path_indices.values(),
            criterion.path_codes.values(),
        ):
            assert path_code.size(0) == path_idx.size(0)
            assert path_code.size(0) <= VOCAB_SIZE
            assert path_idx.size(0) <= VOCAB_SIZE

    def test_hs_forward(self):
        x = torch.randn(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        e = nn.Embedding(VOCAB_SIZE - 1, EMBED_DIM)

        root = build_huffman_tree(WORD_COUNTS)
        criterion = HierarchicalSoftmaxLoss(root, VOCAB_SIZE, context=CONTEXT)
        loss = criterion(x, y, output_embedding=e)

        assert not torch.isnan(loss)

    def test_hs_backward(self):
        x = torch.randn(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        e = nn.Embedding(VOCAB_SIZE - 1, EMBED_DIM)

        root = build_huffman_tree(WORD_COUNTS)
        criterion = HierarchicalSoftmaxLoss(root, VOCAB_SIZE, context=CONTEXT)
        loss = criterion(x, y, output_embedding=e)
        loss.backward()

        assert not torch.isnan(e.weight.grad).any()

    def test_hs_cbow_forward(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))

        root = build_huffman_tree(WORD_COUNTS)
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.CBOW,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
            root=root,
        )
        yhat = model(x)

        criterion = HierarchicalSoftmaxLoss(root, VOCAB_SIZE, context=CONTEXT)
        loss = criterion(yhat, y, output_embedding=model.head.node_embedding)

        assert not torch.isnan(loss)

    def test_hs_sg_forward(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))

        root = build_huffman_tree(WORD_COUNTS)
        model = Word2Vec(
            VOCAB_SIZE,
            EMBED_DIM,
            model_type=Word2VecModelType.SKIP_GRAM,
            loss_type=LossType.HIERARCHICAL_SOFTMAX,
            context=CONTEXT,
            root=root,
        )
        yhat = model(x, NEIGHBOR_SIZE)

        criterion = HierarchicalSoftmaxLoss(root, VOCAB_SIZE, context=CONTEXT)
        loss = criterion(yhat, y, output_embedding=model.head.node_embedding)

        assert not torch.isnan(loss)


class TestUtils:
    def test_word2vec_subsampling(self):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        p = torch.randn(VOCAB_SIZE).softmax(0)
        y = subsampling(x, p, CONTEXT.unk_idx)

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
