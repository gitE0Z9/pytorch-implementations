import pytest
import torch

from torchlake.common.schemas.nlp import NlpContext

from ..models.bilstm_crf import BiLSTMCRF, LinearCRF, LinearCRFLoss

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 10
EMBED_DIM = 8
HIDDEN_DIM = 8
NUM_CLASS = 5
CONTEXT = NlpContext(device="cpu", max_seq_len=SEQ_LEN)


class TestNetwork:
    def test_linear_crf_forward(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)
        mask = y.eq(CONTEXT.padding_idx).int()

        criterion = LinearCRF(NUM_CLASS, CONTEXT)
        y = criterion.forward(x, mask)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN))
        assert not torch.isnan(y).any()


class TestModel:
    @pytest.mark.parametrize(
        "is_training,expected_shape",
        [
            [True, (BATCH_SIZE, SEQ_LEN, NUM_CLASS)],
            [False, (BATCH_SIZE, SEQ_LEN)],
        ],
    )
    def test_bilstm_crf_forward(self, is_training: bool, expected_shape: tuple[int]):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)

        model = BiLSTMCRF(
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
        y = model.forward(x)

        assert y.shape == torch.Size(expected_shape)
        assert not torch.isnan(y).any()


class TestLoss:
    def test_linear_crf_loss_hypotheses_score(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)
        mask = y.eq(CONTEXT.padding_idx).int()

        criterion = LinearCRFLoss()
        loss = criterion.calc_hypotheses_score(x, T, mask)

        assert loss.shape == torch.Size((BATCH_SIZE,))
        assert not torch.isnan(loss).any()

    def test_linear_crf_loss_null_hypothesis_score(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)
        mask = y.eq(CONTEXT.padding_idx).int()

        criterion = LinearCRFLoss()
        loss = criterion.calc_null_hypothesis_score(x, y, T, mask)

        assert loss.shape == torch.Size((BATCH_SIZE,))
        assert not torch.isnan(loss).any()

    def test_linear_crf_loss_forward(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)
        mask = y.eq(CONTEXT.padding_idx).int()

        criterion = LinearCRFLoss()
        loss = criterion.forward(x, y, T)

        assert not torch.isnan(loss)

    def test_linear_crf_loss_backward(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)
        mask = y.eq(CONTEXT.padding_idx).int()

        criterion = LinearCRFLoss()
        loss = criterion.forward(x, y, T)
        loss.backward()
