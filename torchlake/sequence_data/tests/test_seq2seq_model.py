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
    def build_model(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
    ) -> Seq2Seq:
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

        return Seq2Seq(encoder, decoder, context=CONTEXT)

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("early_stopping", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        early_stopping: bool,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.train()
        output = model.loss_forward(x, y, early_stopping=early_stopping)

        if not early_stopping:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            assert output.shape[0] == BATCH_SIZE
            assert output.shape[1] <= SEQ_LEN
            assert output.shape[2] == VOCAB_SIZE

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("topk", [1, 2])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        topk: int,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )
        model.eval()
        output = model.predict(x, topk=topk)

        assert output.size(0) == BATCH_SIZE
        assert output.size(1) <= SEQ_LEN + 1


class TestSeq2SeqWithBahdanauAttention:
    def build_model(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
    ) -> Seq2Seq:
        encoder_factor = 2 if encoder_bidirectional else 1

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
                context_dim=HIDDEN_DIM * encoder_factor,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            BahdanauAttention(
                HIDDEN_DIM * encoder_factor,
                HIDDEN_DIM,
                decoder_bidirectional,
            ),
        )

        return Seq2Seq(encoder, decoder, context=CONTEXT)

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    @pytest.mark.parametrize("early_stopping", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
        early_stopping: bool,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.train()
        output = model.loss_forward(
            x,
            y,
            output_score=output_score,
            early_stopping=early_stopping,
        )

        if output_score:
            output, att = output
            assert att.shape[:2] == torch.Size((BATCH_SIZE, SEQ_LEN))
            if not early_stopping:
                assert att.shape[2] == SEQ_LEN - 1
            else:
                assert att.shape[2] <= SEQ_LEN

        if not early_stopping:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            assert output.shape[0] == BATCH_SIZE
            assert output.shape[1] <= SEQ_LEN
            assert output.shape[2] == VOCAB_SIZE

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    @pytest.mark.parametrize("topk", [1, 2])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
        topk: int,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.eval()
        output = model.predict(x, output_score=output_score, topk=topk)

        if output_score:
            output, att = output
            assert att.shape[:2] == torch.Size((BATCH_SIZE, SEQ_LEN))
            assert att.shape[2] <= SEQ_LEN

        assert output.size(0) == BATCH_SIZE
        assert output.size(1) <= SEQ_LEN + 1


class TestSeq2SeqWithGlobalAttention:
    def build_model(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
    ) -> Seq2Seq:
        encoder_factor = 2 if encoder_bidirectional else 1

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
                context_dim=HIDDEN_DIM * encoder_factor,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            GlobalAttention(
                HIDDEN_DIM * encoder_factor,
                HIDDEN_DIM,
                decoder_bidirectional,
            ),
        )

        return Seq2Seq(encoder, decoder, context=CONTEXT)

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    @pytest.mark.parametrize("early_stopping", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
        early_stopping: bool,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.train()
        output = model.loss_forward(
            x,
            y,
            output_score=output_score,
            early_stopping=early_stopping,
        )

        if output_score:
            output, att = output
            assert att.shape[:2] == torch.Size((BATCH_SIZE, SEQ_LEN))
            if not early_stopping:
                assert att.shape[2] == SEQ_LEN - 1
            else:
                assert att.shape[2] <= SEQ_LEN

        if not early_stopping:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            assert output.shape[0] == BATCH_SIZE
            assert output.shape[1] <= SEQ_LEN
            assert output.shape[2] == VOCAB_SIZE

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    @pytest.mark.parametrize("topk", [1, 2])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
        topk: int,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.eval()
        output = model.predict(x, output_score=output_score, topk=topk)

        if output_score:
            output, att = output
            assert att.shape[:2] == torch.Size((BATCH_SIZE, SEQ_LEN))
            assert att.shape[2] <= SEQ_LEN

        assert output.size(0) == BATCH_SIZE
        assert output.size(1) <= SEQ_LEN + 1


class TestSeq2SeqWithLocalAttention:
    def build_model(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
    ) -> Seq2Seq:
        encoder_factor = 2 if encoder_bidirectional else 1

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
                context_dim=HIDDEN_DIM * encoder_factor,
                num_layers=decoder_num_layers,
                bidirectional=decoder_bidirectional,
                context=CONTEXT,
            ),
            LocalAttention(
                HIDDEN_DIM * encoder_factor,
                HIDDEN_DIM,
                decoder_bidirectional,
                context_size=CONTEXT_SIZE,
            ),
        )

        return Seq2Seq(encoder, decoder, context=CONTEXT)

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    @pytest.mark.parametrize("early_stopping", [True, False])
    def test_loss_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
        early_stopping: bool,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.train()
        output = model.loss_forward(
            x,
            y,
            output_score=output_score,
            early_stopping=early_stopping,
        )

        if output_score:
            output, att = output
            assert att.shape[:2] == torch.Size((BATCH_SIZE, WINDOW_SIZE))
            if not early_stopping:
                assert att.shape[2] == SEQ_LEN - 1
            else:
                assert att.shape[2] <= SEQ_LEN

        if not early_stopping:
            assert output.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        else:
            assert output.shape[0] == BATCH_SIZE
            assert output.shape[1] <= SEQ_LEN
            assert output.shape[2] == VOCAB_SIZE

    @pytest.mark.parametrize("encoder_num_layers,decoder_num_layers", [(1, 1), (2, 2)])
    @pytest.mark.parametrize(
        "encoder_bidirectional,decoder_bidirectional",
        [(True, True), (False, False)],
    )
    @pytest.mark.parametrize("output_score", [True, False])
    @pytest.mark.parametrize("topk", [1, 2])
    def test_predict_forward_shape(
        self,
        encoder_num_layers: int,
        decoder_num_layers: int,
        encoder_bidirectional: bool,
        decoder_bidirectional: bool,
        output_score: bool,
        topk: int,
    ):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = self.build_model(
            encoder_num_layers,
            decoder_num_layers,
            encoder_bidirectional,
            decoder_bidirectional,
        )

        model.eval()
        output = model.predict(x, output_score=output_score, topk=topk)

        if output_score:
            output, att = output
            assert att.shape[:2] == torch.Size((BATCH_SIZE, WINDOW_SIZE))
            assert att.shape[2] <= SEQ_LEN

        assert output.size(0) == BATCH_SIZE
        assert output.size(1) <= SEQ_LEN + 1
