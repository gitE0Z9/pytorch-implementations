import pytest
import torch

from torchlake.common.schemas.nlp import NLPContext

from ..models.bilstm_crf import BiLSTMCRF, LinearCRF, LinearCRFLoss

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 10
EMBED_DIM = 8
HIDDEN_DIM = 8
NUM_CLASS = 5
CONTEXT = NLPContext(device="cpu", max_seq_len=SEQ_LEN)


class TestNetwork:
    @pytest.mark.parametrize("output_score", (True, False))
    def test_linear_crf_forward_shape(self, output_score: bool):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)
        mask = y.eq(CONTEXT.padding_idx).int()

        criterion = LinearCRF(NUM_CLASS, CONTEXT)
        y = criterion(
            x,
            mask,
            output_score=output_score,
        )

        if output_score:
            y, score = y
            assert score.shape == torch.Size((BATCH_SIZE,))
            assert not score.isnan().any()

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN))
        assert not y.isnan().any()


class TestModel:
    @pytest.mark.parametrize(
        "is_training,expected_shape",
        [
            [True, (BATCH_SIZE, SEQ_LEN, NUM_CLASS)],
            [False, (BATCH_SIZE, SEQ_LEN)],
        ],
    )
    def test_bilstm_crf_forward_shape(
        self, is_training: bool, expected_shape: tuple[int]
    ):
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
        y = model(x)

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

    @pytest.mark.parametrize("reduction", ("sum", "mean", None))
    @pytest.mark.parametrize("return_all_loss", (True, False))
    @pytest.mark.parametrize("crf_weight,cross_entroy_weight", ((1, 0), (0, 1), (1, 1)))
    def test_linear_crf_loss_forward(
        self,
        reduction: str | None,
        return_all_loss: bool,
        crf_weight: float,
        cross_entroy_weight: float,
    ):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)

        criterion = LinearCRFLoss(
            crf_weight=crf_weight,
            cross_entroy_weight=cross_entroy_weight,
            context=CONTEXT,
            reduction=reduction,
            return_all_loss=return_all_loss,
        )
        loss = criterion(x, y, T)

        if return_all_loss:
            loss, crf_loss, ce_loss = loss

        if reduction is None:
            assert loss.shape == torch.Size((BATCH_SIZE,))
            assert not loss.isnan().any()
            if return_all_loss:
                assert crf_loss.shape == torch.Size((BATCH_SIZE,))
                assert ce_loss.shape == torch.Size((BATCH_SIZE,))
        else:
            assert not torch.isnan(loss)

    @pytest.mark.parametrize("reduction", ("sum", "mean"))
    @pytest.mark.parametrize("crf_weight,cross_entroy_weight", ((1, 0), (0, 1), (1, 1)))
    def test_linear_crf_loss_backward(
        self,
        reduction: str | None,
        crf_weight: float,
        cross_entroy_weight: float,
    ):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, NUM_CLASS)
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, SEQ_LEN))
        T = torch.rand(NUM_CLASS, NUM_CLASS)
        T.requires_grad_(True)

        criterion = LinearCRFLoss(
            crf_weight=crf_weight,
            cross_entroy_weight=cross_entroy_weight,
            context=CONTEXT,
            reduction=reduction,
        )
        loss = criterion(x, y, T)
        loss.backward()
