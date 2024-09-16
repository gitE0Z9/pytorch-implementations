from unittest import TestCase
import torch
from torchlake.common.schemas.nlp import NlpContext

from ..models.vlbl.model import VLBL, IVLBL
from ..models.vlbl.loss import NCE

BATCH_SIZE = 2
VOCAB_SIZE = 16
CONTEXT_SIZE = 5
EMBED_SIZE = 8
NEIGHBOR_SIZE = CONTEXT_SIZE - 1
SUBSEQ_LEN = 256 - NEIGHBOR_SIZE
NEGATIVE_RATIO = 5
CONTEXT = NlpContext(device="cpu")
WORD_FREQS = torch.rand((VOCAB_SIZE))


class TestVLBL(TestCase):
    def test_vlbl_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        y = model.forward(context, gram)

        self.assertEqual(
            y.shape,
            torch.Size((BATCH_SIZE, 1, SUBSEQ_LEN)),
        )

    def test_ivlbl_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        y = model.forward(gram, context)

        self.assertEqual(
            y.shape,
            torch.Size((BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN)),
        )


class TestNCE(TestCase):
    def setUp(self) -> None:
        self.criterion = NCE(WORD_FREQS, context=CONTEXT)

    def test_get_distribution_shape(self):
        self.assertEqual(
            self.criterion.distribution.shape,
            torch.Size((VOCAB_SIZE,)),
        )

    def test_sample_shape(self):
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        y = self.criterion.sample(context)

        self.assertEqual(
            y.shape,
            torch.Size((BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN * NEGATIVE_RATIO)),
        )

    def test_vlbl_forward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        yhat = model.forward(context, gram)
        loss = self.criterion.forward(model, context, gram, yhat)

        assert not torch.isnan(loss)

    def test_ivlbel_forward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        yhat = model.forward(gram, context)
        loss = self.criterion.forward(model, gram, context, yhat)

        assert not torch.isnan(loss)

    def test_vlbl_backward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = VLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        yhat = model.forward(context, gram)
        loss = self.criterion.forward(model, context, gram, yhat)
        loss.backward()

        assert not torch.isnan(model.word_embed.weight.grad).any()

    def test_ivlbel_backward(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1, SUBSEQ_LEN))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NEIGHBOR_SIZE, SUBSEQ_LEN))
        model = IVLBL(VOCAB_SIZE, EMBED_SIZE, NEIGHBOR_SIZE, context=CONTEXT)

        yhat = model.forward(gram, context)
        loss = self.criterion.forward(model, gram, context, yhat)
        loss.backward()

        assert not torch.isnan(model.context_embed.weight.grad).any()
