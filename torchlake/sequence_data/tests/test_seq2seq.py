import pytest
import torch
from torchlake.common.schemas.nlp import NlpContext

from ..models.seq2seq.model import Seq2Seq
from ..models.seq2seq.network import (
    Seq2SeqDecoder,
    Seq2SeqEncoder,
    Seq2SeqAttentionEncoder,
    GlobalAttention,
    LocalAttention,
)


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_forward_shape_global_attention(name: str, bidirectional: bool, factor: int):
    hs = torch.rand((2, 256, 16 * factor))
    # D * num_layers, B, h
    ht = torch.rand((factor, 2, 16))
    model = GlobalAttention(16, bidirectional=bidirectional)
    y, a = model(hs, ht)

    assert y.shape == torch.Size((factor, 2, 16))
    assert a.shape == torch.Size((factor, 2, 256))


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_prediction_position_local_attention(
    name: str, bidirectional: bool, factor: int
):
    context_size = 2
    window_size = 2 * context_size + 1
    ht = torch.rand((factor, 2, 16))
    model = LocalAttention(16, context_size=context_size, bidirectional=bidirectional)
    pt, position = model.get_predicted_position(ht, 256)

    assert pt.shape == torch.Size((factor, 2, 1))
    assert position.shape == torch.Size((factor, 2, window_size))


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_kernel_local_attention(name: str, bidirectional: bool, factor: int):
    context_size = 2
    window_size = 2 * context_size + 1
    pt = torch.rand((factor, 2, 1))
    positions = torch.randint(0, 256, (factor, 2, window_size))
    model = LocalAttention(16, context_size=context_size, bidirectional=bidirectional)
    kernel = model.get_kernel(pt, positions, 16)

    assert kernel.shape == torch.Size((factor, 2, window_size))


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_subsample_source_hidden_state_local_attention(
    name: str, bidirectional: bool, factor: int
):
    context_size = 2
    window_size = 2 * context_size + 1
    hs = torch.rand((2, 256, factor * 16))
    positions = torch.randint(-context_size + 1, 256, (factor, 2, window_size))
    model = LocalAttention(16, context_size=context_size, bidirectional=bidirectional)
    kernel = model.subsample_source_hidden_state(hs, positions)

    assert kernel.shape == torch.Size((factor, 2, window_size, 16))


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_forward_shape_local_attention(name: str, bidirectional: bool, factor: int):
    context_size = 2
    window_size = 2 * context_size + 1
    hs = torch.rand((2, 256, factor * 16))
    ht = torch.rand((factor, 2, 16))
    model = LocalAttention(16, context_size=context_size, bidirectional=bidirectional)
    y, a = model(hs, ht)

    assert y.shape == torch.Size((factor, 2, 16))
    assert a.shape == torch.Size((factor, 2, window_size))


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_forward_shape_encoder(name: str, bidirectional: bool, factor: int):
    x = torch.randint(0, 100, (2, 256))
    model = Seq2SeqEncoder(100, 16, 16, bidirectional=bidirectional)
    hs, (ht, ct) = model(x)

    assert hs.shape == torch.Size((2, 256, 16 * factor))
    assert ht.shape == torch.Size((factor, 2, 16))
    assert ct.shape == torch.Size((factor, 2, 16))


@pytest.mark.parametrize(
    "name,num_layers,bidirectional,factor",
    [
        ["single-layer-unidirectional", 1, False, 1],
        ["single-layer-bidirectional", 1, True, 2],
        ["multiple-layer-unidirectional", 2, False, 1],
        ["multiple-layer-bidirectional", 2, True, 2],
    ],
)
def test_forward_shape_attention_encoder(
    name: str,
    num_layers: int,
    bidirectional: bool,
    factor: int,
):
    x = torch.randint(0, 100, (2, 256))
    model = Seq2SeqAttentionEncoder(
        100,
        16,
        16,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    hs, (ht, ct) = model(x)

    assert hs.shape == torch.Size((2, 256, 16 * factor * num_layers))
    assert ht.shape == torch.Size((factor * num_layers, 2, 16))
    assert ct.shape == torch.Size((factor * num_layers, 2, 16))


@pytest.mark.parametrize(
    "name,bidirectional,factor",
    [
        ["unidirectional", False, 1],
        ["bidirectional", True, 2],
    ],
)
def test_forward_shape_decoder(name: str, bidirectional: bool, factor: int):
    x = torch.randint(0, 100, (2, 256))
    h, c = torch.rand((factor, 2, 16)), torch.rand((factor, 2, 16))
    model = Seq2SeqDecoder(100, 16, 16, 100, bidirectional=bidirectional)
    y, (ht, ct) = model(x, h, c)

    assert y.shape == torch.Size((2, 100))
    assert ht.shape == torch.Size((factor, 2, 16))
    assert ct.shape == torch.Size((factor, 2, 16))


@pytest.mark.parametrize(
    "name,num_layers,bidirectional,factor",
    [
        ["single-layer-unidirectional", 1, False, 1],
        ["single-layer-bidirectional", 1, True, 2],
        ["multiple-layer-unidirectional", 2, False, 1],
        ["multiple-layer-bidirectional", 2, True, 2],
    ],
)
def test_forward_shape_model(
    name: str,
    num_layers: int,
    bidirectional: bool,
    factor: int,
):
    x = torch.randint(0, 100, (2, 256))
    y = torch.randint(0, 100, (2, 256))
    encoder = Seq2SeqEncoder(
        100,
        16,
        16,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    decoder = Seq2SeqDecoder(
        100,
        16,
        16,
        100,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model = Seq2Seq(encoder, decoder, context=NlpContext(device="cpu"))
    output = model(x, y)

    assert output.shape == torch.Size((2, 256, 100))


@pytest.mark.parametrize(
    "name,num_layers,bidirectional,factor",
    [
        ["single-layer-unidirectional", 1, False, 1],
        ["single-layer-bidirectional", 1, True, 2],
        ["multiple-layer-unidirectional", 2, False, 1],
        ["multiple-layer-bidirectional", 2, True, 2],
    ],
)
def test_attend_hidden_state_shape_model(
    name: str,
    num_layers: int,
    bidirectional: bool,
    factor: int,
):
    hs = torch.rand((2, 256, 16 * factor))
    ht = torch.rand((factor, 2, 16))
    ct = torch.rand((factor, 2, 16))
    encoder = Seq2SeqEncoder(
        100,
        16,
        16,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    decoder = Seq2SeqDecoder(
        100,
        16,
        16,
        100,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model = Seq2Seq(
        encoder,
        decoder,
        GlobalAttention(16, bidirectional=bidirectional),
        context=NlpContext(device="cpu"),
    )

    ht, ct = model.attend_hidden_state(hs, (ht, ct))

    assert ht.shape == torch.Size((factor, 2, 16))
    assert ct.shape == torch.Size((factor, 2, 16))


@pytest.mark.parametrize(
    "name,num_layers,bidirectional,factor",
    [
        ["single-layer-unidirectional", 1, False, 1],
        ["single-layer-bidirectional", 1, True, 2],
        ["multiple-layer-unidirectional", 2, False, 1],
        ["multiple-layer-bidirectional", 2, True, 2],
    ],
)
def test_forward_shape_model_global_attention(
    name: str,
    num_layers: int,
    bidirectional: bool,
    factor: int,
):
    x = torch.randint(0, 100, (2, 256))
    y = torch.randint(0, 100, (2, 256))
    encoder = Seq2SeqAttentionEncoder(
        100,
        16,
        16,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    decoder = Seq2SeqDecoder(
        100,
        16,
        16,
        100,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model = Seq2Seq(
        encoder,
        decoder,
        GlobalAttention(16, num_layers=num_layers, bidirectional=bidirectional),
        context=NlpContext(device="cpu"),
    )
    output = model(x, y)

    assert output.shape == torch.Size((2, 256, 100))


@pytest.mark.parametrize(
    "name,num_layers,bidirectional,factor",
    [
        ["single-layer-unidirectional", 1, False, 1],
        ["single-layer-bidirectional", 1, True, 2],
        ["multiple-layer-unidirectional", 2, False, 1],
        ["multiple-layer-bidirectional", 2, True, 2],
    ],
)
def test_forward_shape_model_local_attention(
    name: str,
    num_layers: int,
    bidirectional: bool,
    factor: int,
):
    x = torch.randint(0, 100, (2, 256))
    y = torch.randint(0, 100, (2, 256))
    encoder = Seq2SeqAttentionEncoder(
        100,
        16,
        16,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    decoder = Seq2SeqDecoder(
        100,
        16,
        16,
        100,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model = Seq2Seq(
        encoder,
        decoder,
        LocalAttention(16, num_layers=num_layers, bidirectional=bidirectional),
        context=NlpContext(device="cpu"),
    )
    output = model(x, y)

    assert output.shape == torch.Size((2, 256, 100))
