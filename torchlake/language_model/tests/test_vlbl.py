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
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        y = model(context, gram)

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN, 1))

    def test_ivlbl_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        y = model(gram, context)

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))


class TestLoss:
    def test_nce_get_distribution_shape(self):
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        assert criterion.distribution.shape == torch.Size((VOCAB_SIZE,))

    @pytest.mark.parametrize("replacement", (True, False))
    def test_nce_sample_shape(self, replacement: bool):
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        criterion = NCE(WORD_FREQS, replacement=replacement, context=CONTEXT)
        y = criterion.sample(context)

        assert y.shape == torch.Size(
            (BATCH_SIZE * SUBSEQ_LEN * NEGATIVE_RATIO, NEIGHBOR_SIZE)
        )

    def test_nce_vlbl_forward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model(context, gram)
        loss = criterion(yhat, context, gram, model)

        assert not torch.isnan(loss)

    def test_nce_vlbl_backward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model(context, gram)
        loss = criterion(yhat, context, gram, model)
        loss.backward()

        assert not torch.isnan(model.input_embedding.weight.grad).any()

    def test_nce_ivlbl_forward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model(gram, context)
        loss = criterion(yhat, gram, context, model)

        assert not torch.isnan(loss)

    def test_nce_ivlbl_backward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)
        criterion = NCE(WORD_FREQS, context=CONTEXT)

        yhat = model(gram, context)
        loss = criterion(yhat, gram, context, model)
        loss.backward()

        assert not torch.isnan(model.output_embedding.weight.grad).any()
