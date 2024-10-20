import pytest
import torch

from ..models.bilstm_crf import BiLSTM_CRF, LinearCRF, LinearCRFLoss
from .constants import (
    BATCH_SIZE,
    CONTEXT,
    VOCAB_SIZE,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_CLASS,
    SEQ_LEN,
)


class TestLinearCRF:
    def setUp(self):
        self.x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        self.y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        self.T = torch.rand(NUM_CLASS, NUM_CLASS)
        self.T.requires_grad_(True)
        self.mask = self.y.eq(CONTEXT.padding_idx).int()

    def test_forward(self):
        self.setUp()

        criterion = LinearCRF(NUM_CLASS, CONTEXT)
        y = criterion.forward(self.x, self.mask)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN))
        assert not torch.isnan(y).any()


class TestBiLSTM_CRF:
    def setUp(self):
        self.x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        self.y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        self.T = torch.rand(NUM_CLASS, NUM_CLASS)
        self.T.requires_grad_(True)
        self.mask = self.y.eq(CONTEXT.padding_idx).int()

    @pytest.mark.parametrize(
        "is_training,expected_shape",
        [
            [True, (BATCH_SIZE, SEQ_LEN, NUM_CLASS)],
            [False, (BATCH_SIZE, SEQ_LEN)],
        ],
    )
    def test_forward(self, is_training: bool, expected_shape: tuple[int]):
        self.setUp()

        model = BiLSTM_CRF(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            NUM_CLASS,
            context=CONTEXT,
        )
        if is_training:
            model.train()
        else:
            model.eval()
        y = model.forward(self.x, self.mask)

        assert y.shape == torch.Size(expected_shape)
        assert not torch.isnan(y).any()


class TestLinearCRFLoss:
    def setUp(self):
        self.x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        self.y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        self.T = torch.rand(NUM_CLASS, NUM_CLASS)
        self.T.requires_grad_(True)
        self.mask = self.y.eq(CONTEXT.padding_idx).int()

    def test_hypotheses_score(self):
        self.setUp()

        criterion = LinearCRFLoss()
        loss = criterion.calc_hypotheses_score(self.x, self.T, self.mask)

        assert loss.shape == torch.Size((BATCH_SIZE, NUM_CLASS))
        assert not torch.isnan(loss).any()

    def test_null_hypothesis_score(self):
        self.setUp()

        criterion = LinearCRFLoss()
        loss = criterion.calc_null_hypothesis_score(self.x, self.y, self.T, self.mask)

        assert loss.shape == torch.Size((BATCH_SIZE,))
        assert not torch.isnan(loss).any()

    def test_forward(self):
        self.setUp()

        criterion = LinearCRFLoss()
        loss = criterion.forward(self.x, self.y, self.T)

        assert not torch.isnan(loss)

    def test_backward(self):
        self.setUp()

        criterion = LinearCRFLoss()
        loss = criterion.forward(self.x, self.y, self.T)
        loss.backward()
