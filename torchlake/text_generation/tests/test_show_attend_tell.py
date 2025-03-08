from functools import partial
import pytest
import torch

from torchlake.common.schemas.nlp import NlpContext

from torchlake.sequence_data.models.base.rnn_generator import RNNGenerator
from ..models.show_attend_tell import (
    DoublyStochasticAttentionLoss,
    HardAttention,
    SoftAttention,
    ShowAttendTell,
)
from torchlake.sequence_data.models.lstm import LSTMDiscriminator
from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 32
EMBED_DIM = 4
ENCODE_DIM = 512
DECODE_DIM = 8
CONTEXT = NlpContext(device="cpu", max_seq_len=SEQ_LEN)
extractor = VGGFeatureExtractor("vgg16", "relu", False)
extractor.forward = partial(extractor.forward, target_layer_names=["5_3"])
NUM_PATCH = 196


class TestNetwork:

    @pytest.mark.parametrize("bidirectional,D", [(True, 2), (False, 1)])
    def test_forward_shape_soft_attention(self, bidirectional: bool, D: int):
        os = torch.rand(BATCH_SIZE, SEQ_LEN, ENCODE_DIM)
        ht = torch.rand(D, BATCH_SIZE, DECODE_DIM)

        model = SoftAttention(ENCODE_DIM, DECODE_DIM, bidirectional)

        y, a = model(os, ht)

        assert y.shape == torch.Size((BATCH_SIZE, 1, ENCODE_DIM))
        assert a.shape == torch.Size((BATCH_SIZE, SEQ_LEN))

    @pytest.mark.parametrize("bidirectional,D", [(True, 2), (False, 1)])
    def test_forward_shape_hard_attention(self, bidirectional: bool, D: int):
        os = torch.rand(BATCH_SIZE, SEQ_LEN, ENCODE_DIM)
        ht = torch.rand(D, BATCH_SIZE, DECODE_DIM)

        model = HardAttention(ENCODE_DIM, DECODE_DIM, bidirectional)

        y, a = model(os, ht)

        assert y.shape == torch.Size((BATCH_SIZE, 1, ENCODE_DIM))
        assert a.shape == torch.Size((BATCH_SIZE, SEQ_LEN))


class TestModel:
    @pytest.mark.parametrize(
        "is_train,expected_shape",
        [(True, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)), (False, (BATCH_SIZE, SEQ_LEN))],
    )
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_forward_shape_soft_attention(
        self,
        is_train: bool,
        expected_shape: tuple,
        bidirectional: bool,
    ):
        x = torch.rand(BATCH_SIZE, 3, 224, 224)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        model = ShowAttendTell(
            backbone=extractor,
            decoder=RNNGenerator(
                model=LSTMDiscriminator(
                    VOCAB_SIZE,
                    EMBED_DIM,
                    DECODE_DIM,
                    VOCAB_SIZE,
                    context_dim=ENCODE_DIM,
                    bidirectional=bidirectional,
                    context=CONTEXT,
                ),
                attention=SoftAttention(
                    ENCODE_DIM,
                    DECODE_DIM,
                    decoder_bidirectional=bidirectional,
                ),
            ),
            context=CONTEXT,
        )

        if is_train:
            model.train()
            y, a = model(x, y, early_stopping=True)
            assert y.size(0) == expected_shape[0]
            assert y.size(1) <= expected_shape[1]
            assert y.size(2) == expected_shape[2]

            assert a.size(0) == BATCH_SIZE
            assert a.size(1) == NUM_PATCH
            assert a.size(2) <= SEQ_LEN
        else:
            model.eval()
            y = model(x)
            assert y.size(0) == BATCH_SIZE
            assert y.size(1) <= SEQ_LEN + 1

    @pytest.mark.parametrize(
        "is_train,expected_shape",
        [(True, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)), (False, (BATCH_SIZE, SEQ_LEN))],
    )
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_forward_shape_hard_attention(
        self,
        is_train: bool,
        expected_shape: tuple,
        bidirectional: bool,
    ):
        x = torch.rand(BATCH_SIZE, 3, 224, 224)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        model = ShowAttendTell(
            backbone=extractor,
            decoder=RNNGenerator(
                model=LSTMDiscriminator(
                    VOCAB_SIZE,
                    EMBED_DIM,
                    DECODE_DIM,
                    VOCAB_SIZE,
                    context_dim=ENCODE_DIM,
                    bidirectional=bidirectional,
                    context=CONTEXT,
                ),
                attention=HardAttention(
                    ENCODE_DIM,
                    DECODE_DIM,
                    decoder_bidirectional=bidirectional,
                ),
            ),
            context=CONTEXT,
        )

        if is_train:
            model.train()
            y, a = model(x, y, early_stopping=True)
            assert y.size(0) == expected_shape[0]
            assert y.size(1) <= expected_shape[1]
            assert y.size(2) == expected_shape[2]

            assert a.size(0) == BATCH_SIZE
            assert a.size(1) == NUM_PATCH
            assert a.size(2) <= SEQ_LEN
        else:
            model.eval()
            y = model(x)
            assert y.size(0) == BATCH_SIZE
            assert y.size(1) <= SEQ_LEN + 1


class TestLoss:
    def test_forward(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE).transpose(1, 2)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        a = torch.rand(BATCH_SIZE, NUM_PATCH, SEQ_LEN)
        criterion = DoublyStochasticAttentionLoss()

        loss = criterion(x, y, a)
        assert not torch.isnan(loss)

    def test_backward(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE).transpose(1, 2).requires_grad_()
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        a = torch.rand(BATCH_SIZE, NUM_PATCH, SEQ_LEN).requires_grad_()
        criterion = DoublyStochasticAttentionLoss()

        loss = criterion(x, y, a)
        loss.backward()
