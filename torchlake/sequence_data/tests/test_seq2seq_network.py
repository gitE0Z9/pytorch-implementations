import pytest
import torch

from ..models.seq2seq.network import BahdanauAttention, GlobalAttention, LocalAttention

BATCH_SIZE = 2
SEQ_LEN = 32


class TestBahdanauAttention:

    @pytest.mark.parametrize("encode_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "decoder_bidirectional,decoder_factor", [(True, 2), (False, 1)]
    )
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_forward_shape(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor

        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encode_dim))
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))
        model = BahdanauAttention(
            encode_dim,
            decoder_hidden_dim,
            decoder_bidirectional,
        )
        c, a = model(hs, ht)

        assert c.shape == torch.Size((BATCH_SIZE, 1, encode_dim))
        assert a.shape == torch.Size((BATCH_SIZE, SEQ_LEN))


class TestGlobalAttention:

    @pytest.mark.parametrize("encode_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "decoder_bidirectional,decoder_factor", [(True, 2), (False, 1)]
    )
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_forward_shape(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor

        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encode_dim))
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))
        model = GlobalAttention(
            encode_dim,
            decoder_hidden_dim,
            decoder_bidirectional,
        )
        c, a = model(hs, ht)

        assert c.shape == torch.Size((BATCH_SIZE, 1, encode_dim))
        assert a.shape == torch.Size((BATCH_SIZE, SEQ_LEN))


class TestLocalAttention:

    @pytest.mark.parametrize("encode_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "decoder_bidirectional,decoder_factor", [(True, 2), (False, 1)]
    )
    def test_prediction_position(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool,
        decoder_factor: int,
    ):
        context_size = 2
        window_size = 2 * context_size + 1
        ht = torch.rand((BATCH_SIZE, decoder_hidden_dim * decoder_factor))
        model = LocalAttention(
            encode_dim,
            decoder_hidden_dim,
            decoder_bidirectional,
            context_size=context_size,
        )
        pos, window = model.get_predicted_position(ht, SEQ_LEN)

        assert pos.shape == torch.Size((BATCH_SIZE, 1))
        assert window.shape == torch.Size((BATCH_SIZE, window_size))

    @pytest.mark.parametrize("encode_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_bidirectional", [True, False])
    def test_kernel_shape(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool,
    ):
        context_size = 2
        window_size = 2 * context_size + 1
        pos = torch.rand((BATCH_SIZE, 1)) * SEQ_LEN
        window = torch.randint(0, SEQ_LEN, (BATCH_SIZE, window_size))

        model = LocalAttention(
            encode_dim,
            decoder_hidden_dim,
            decoder_bidirectional,
            context_size=context_size,
        )
        kernel = model.get_kernel(pos, window)

        assert kernel.shape == torch.Size((BATCH_SIZE, window_size))

    @pytest.mark.parametrize("encode_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_bidirectional", [True, False])
    def test_subsample_source_hidden_state(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool,
    ):
        context_size = 2
        window_size = 2 * context_size + 1
        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encode_dim))
        window = torch.randint(
            -context_size,
            SEQ_LEN + context_size,
            (BATCH_SIZE, window_size),
        )

        model = LocalAttention(
            encode_dim,
            decoder_hidden_dim,
            decoder_bidirectional,
            context_size=context_size,
        )
        hs = model.subsample_source_hidden_state(hs, window)

        assert hs.shape == torch.Size((BATCH_SIZE, window_size, encode_dim))

    @pytest.mark.parametrize("encode_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "decoder_bidirectional,decoder_factor", [(True, 2), (False, 1)]
    )
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_forward_shape(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor
        context_size = 2
        window_size = 2 * context_size + 1
        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encode_dim))
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))

        model = LocalAttention(
            encode_dim,
            decoder_hidden_dim,
            decoder_bidirectional,
            context_size=context_size,
        )
        c, a = model(hs, ht)

        assert c.shape == torch.Size((BATCH_SIZE, 1, encode_dim))
        assert a.shape == torch.Size((BATCH_SIZE, window_size))
