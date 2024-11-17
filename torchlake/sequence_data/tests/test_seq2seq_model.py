import pytest
import torch

from torchlake.common.schemas.nlp import NlpContext

from ..models.lstm.model import LSTMDiscriminator
from ..models.base.rnn_generator import RNNGenerator
from ..models.seq2seq.model import Seq2Seq
from ..models.seq2seq.network import (
    BahdanauAttention,
    GlobalAttention,
    LocalAttention,
)

SEQ_LEN = 16
VOCAB_SIZE = 10
BATCH_SIZE = 4
EMBED_DIM = 8
HIDDEN_DIM = 16
CONTEXT = NlpContext(max_seq_len=SEQ_LEN, device="cpu")
CONTEXT_SIZE = 2
WINDOW_SIZE = 5


class TestSeq2Seq:
    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            )
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.train()
        output = model.loss_forward(x, y)

        assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            )
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.eval()
        output = model.predict(x)

        assert output.size(0) == BATCH_SIZE
        assert output.size(1) <= SEQ_LEN + 1


class TestSeq2SeqWithBahdanauAttention:
    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
    ):
        D = decoder_num_layers * (2 if decoder_bidirectional else 1)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            BahdanauAttention(HIDDEN_DIM, HIDDEN_DIM, encoder_bidirectional),
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.train()
        output = model.loss_forward(x, y, output_score=output_score)

        if not output_score:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            o, a = output
            assert o.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
            assert a.shape[:3] == torch.Size((D, BATCH_SIZE, SEQ_LEN))
            assert a.shape[3] <= SEQ_LEN

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
    ):
        D = decoder_num_layers * (2 if decoder_bidirectional else 1)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            BahdanauAttention(HIDDEN_DIM, HIDDEN_DIM, encoder_bidirectional),
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.eval()
        output = model.predict(x, output_score=output_score)

        if not output_score:
            assert output.size(0) == BATCH_SIZE
            assert output.size(1) <= SEQ_LEN + 1
        else:
            o, a = output
            assert o.size(0) == BATCH_SIZE
            assert o.size(1) <= SEQ_LEN + 1
            assert a.shape[:3] == torch.Size((D, BATCH_SIZE, SEQ_LEN))
            assert a.shape[3] <= SEQ_LEN


class TestSeq2SeqWithGlobalAttention:
    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
    ):
        D = decoder_num_layers * (2 if decoder_bidirectional else 1)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            GlobalAttention(HIDDEN_DIM, HIDDEN_DIM, encoder_bidirectional),
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.train()
        output = model.loss_forward(x, y, output_score=output_score)

        if not output_score:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            o, a = output
            assert o.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
            assert a.shape[:3] == torch.Size((D, BATCH_SIZE, SEQ_LEN))
            assert a.shape[3] <= SEQ_LEN

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
    ):
        D = decoder_num_layers * (2 if decoder_bidirectional else 1)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            GlobalAttention(HIDDEN_DIM, HIDDEN_DIM, encoder_bidirectional),
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.eval()
        output = model.predict(x, output_score=output_score)

        if not output_score:
            assert output.size(0) == BATCH_SIZE
            assert output.size(1) <= SEQ_LEN + 1
        else:
            o, a = output
            assert o.size(0) == BATCH_SIZE
            assert o.size(1) <= SEQ_LEN + 1
            assert a.shape[:3] == torch.Size((D, BATCH_SIZE, SEQ_LEN))
            assert a.shape[3] <= SEQ_LEN


class TestSeq2SeqWithLocalAttention:
    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
    ):
        D = decoder_num_layers * (2 if decoder_bidirectional else 1)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            LocalAttention(HIDDEN_DIM, HIDDEN_DIM, encoder_bidirectional, CONTEXT_SIZE),
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.train()
        output = model.loss_forward(x, y, output_score=output_score)

        if not output_score:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            o, a = output
            assert o.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
            assert a.shape[:3] == torch.Size((D, BATCH_SIZE, WINDOW_SIZE))
            assert a.shape[3] <= SEQ_LEN

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
    ):
        D = decoder_num_layers * (2 if decoder_bidirectional else 1)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = LSTMDiscriminator(
            VOCAB_SIZE,
            EMBED_DIM,
            HIDDEN_DIM,
            num_layers=encoder_num_layers,
            bidirectional=encoder_bidirectional,
            drop_fc=True,
            context=CONTEXT,
        )
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                EMBED_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            LocalAttention(HIDDEN_DIM, HIDDEN_DIM, encoder_bidirectional, CONTEXT_SIZE),
        )
        model = Seq2Seq(encoder, decoder, context=CONTEXT)
        model.eval()
        output = model.predict(x, output_score=output_score)

        if not output_score:
            assert output.size(0) == BATCH_SIZE
            assert output.size(1) <= SEQ_LEN + 1
        else:
            o, a = output
            assert o.size(0) == BATCH_SIZE
            assert o.size(1) <= SEQ_LEN + 1
            assert a.shape[:3] == torch.Size((D, BATCH_SIZE, WINDOW_SIZE))
            assert a.shape[3] <= SEQ_LEN
