import pytest
import torch

from ..models.seq2seq.network import BahdanauAttention, GlobalAttention, LocalAttention

BATCH_SIZE = 2
SEQ_LEN = 32


class TestBahdanauAttention:

    @pytest.mark.parametrize("encoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "encoder_bidirectional,encoder_factor",
        [(True, 2), (False, 1)],
    )
    @pytest.mark.parametrize("decoder_factor", [2, 1])
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_forward_shape(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool,
        encoder_factor: int,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor

        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encoder_factor * encoder_hidden_dim))
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))
        model = BahdanauAttention(
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_bidirectional,
        )
        c, a = model(hs, ht)

        assert c.shape == torch.Size((D, BATCH_SIZE, decoder_hidden_dim))
        assert a.shape == torch.Size((D, BATCH_SIZE, SEQ_LEN))


class TestGlobalAttention:

    @pytest.mark.parametrize("encoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "encoder_bidirectional,encoder_factor",
        [(True, 2), (False, 1)],
    )
    @pytest.mark.parametrize("decoder_factor", [2, 1])
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_forward_shape(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool,
        encoder_factor: int,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor

        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encoder_factor * encoder_hidden_dim))
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))
        model = GlobalAttention(
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_bidirectional,
        )
        c, a = model(hs, ht)

        assert c.shape == torch.Size((D, BATCH_SIZE, decoder_hidden_dim))
        assert a.shape == torch.Size((D, BATCH_SIZE, SEQ_LEN))


class TestLocalAttention:

    @pytest.mark.parametrize("encoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("encoder_bidirectional", [True, False])
    @pytest.mark.parametrize("decoder_factor", [2, 1])
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_prediction_position(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor
        context_size = 2
        window_size = 2 * context_size + 1
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))
        model = LocalAttention(
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_bidirectional,
            context_size=context_size,
        )
        pos, window = model.get_predicted_position(ht, SEQ_LEN)

        assert pos.shape == torch.Size((D, BATCH_SIZE, 1))
        assert window.shape == torch.Size((D, BATCH_SIZE, window_size))

    @pytest.mark.parametrize("encoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("encoder_bidirectional", [True, False])
    @pytest.mark.parametrize("decoder_factor", [2, 1])
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_kernel_shape(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor
        context_size = 2
        window_size = 2 * context_size + 1
        pos = torch.rand((D, BATCH_SIZE, 1)) * SEQ_LEN
        window = torch.randint(0, SEQ_LEN, (D, BATCH_SIZE, window_size))

        model = LocalAttention(
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_bidirectional,
            context_size=context_size,
        )
        kernel = model.get_kernel(pos, window)

        assert kernel.shape == torch.Size((D, BATCH_SIZE, window_size))

    @pytest.mark.parametrize("encoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "encoder_bidirectional,encoder_factor",
        [(True, 2), (False, 1)],
    )
    @pytest.mark.parametrize("decoder_factor", [2, 1])
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_subsample_source_hidden_state(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool,
        encoder_factor: int,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor
        context_size = 2
        window_size = 2 * context_size + 1
        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encoder_factor * encoder_hidden_dim))
        window = torch.randint(
            -context_size,
            SEQ_LEN + context_size,
            (D, BATCH_SIZE, window_size),
        )

        model = LocalAttention(
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_bidirectional,
            context_size=context_size,
        )
        hs = model.subsample_source_hidden_state(hs, window)

        assert hs.shape == torch.Size(
            (D, BATCH_SIZE, window_size, encoder_factor * encoder_hidden_dim)
        )

    @pytest.mark.parametrize("encoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize("decoder_hidden_dim", [16, 32])
    @pytest.mark.parametrize(
        "encoder_bidirectional,encoder_factor",
        [(True, 2), (False, 1)],
    )
    @pytest.mark.parametrize("decoder_factor", [2, 1])
    @pytest.mark.parametrize("decoder_num_layers", [1, 2])
    def test_forward_shape(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool,
        encoder_factor: int,
        decoder_factor: int,
        decoder_num_layers: int,
    ):
        D = decoder_num_layers * decoder_factor
        context_size = 2
        window_size = 2 * context_size + 1
        hs = torch.rand((BATCH_SIZE, SEQ_LEN, encoder_factor * encoder_hidden_dim))
        ht = torch.rand((D, BATCH_SIZE, decoder_hidden_dim))

        model = LocalAttention(
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_bidirectional,
            context_size=context_size,
        )
        y, a = model(hs, ht)

        assert y.shape == torch.Size((D, BATCH_SIZE, decoder_hidden_dim))
        assert a.shape == torch.Size((D, BATCH_SIZE, window_size))
