import torch
import pytest

from torchlake.common.schemas.nlp import NLPContext

from ..models.vlbl.loss import NCE
from ..models.vlbl.model import IVLBL, VLBL

BATCH_SIZE = 2
VOCAB_SIZE = 16
CONTEXT_SIZE = 5
EMBED_SIZE = 8
NEIGHBOR_SIZE = CONTEXT_SIZE - 1
SUBSEQ_LEN = 256 - NEIGHBOR_SIZE
NEGATIVE_RATIO = 5
CONTEXT = NLPContext(device="cpu")
WORD_FREQS = torch.rand((VOCAB_SIZE))


class TestModel:
    def test_vlbl_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        y = model.forward(context, gram)

        assert y.shape == torch.Size((BATCH_SIZE, 1, SUBSEQ_LEN))

    def test_ivlbl_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        y = model.forward(gram, context)

        assert y.shape == torch.Size((BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))


class TestLoss:
    def test_nce_get_distribution_shape(self):
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        assert criterion.distribution.shape == torch.Size((VOCAB_SIZE,))

    @pytest.mark.parametrize("replacement", (True, False))
    def test_nce_sample_shape(self, replacement: bool):
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        criterion = NCE(WORD_FREQS, replacement=replacement, context=CONTEXT)
        y = criterion.sample(context)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN * NEGATIVE_RATIO)
        )

    def test_nce_vlbl_forward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model.forward(context, gram)
        loss = criterion.forward(model, context, gram, yhat)

        assert not torch.isnan(loss)

    def test_nce_ivlbl_forward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model.forward(gram, context)
        loss = criterion.forward(model, gram, context, yhat)

        assert not torch.isnan(loss)

    def test_nce_vlbl_backward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model.forward(context, gram)
        loss = criterion.forward(model, context, gram, yhat)
        loss.backward()

        assert not torch.isnan(model.word_embed.weight.grad).any()

    def test_nce_ivlbl_backward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model.forward(gram, context)
        loss = criterion.forward(model, gram, context, yhat)
        loss.backward()

        assert not torch.isnan(model.context_embed.weight.grad).any()
