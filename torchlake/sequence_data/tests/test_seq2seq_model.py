import pytest
import torch

from torchlake.common.schemas.nlp import NlpContext

from ..models.seq2seq.model import Seq2Seq
from ..models.seq2seq.network import (
    GlobalAttention,
    LocalAttention,
    Seq2SeqAttentionEncoder,
    Seq2SeqDecoder,
    Seq2SeqEncoder,
)


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
    output = model.loss_forward(x, y)

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
    output = model.loss_forward(x, y)

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
    output = model.loss_forward(x, y)

    assert output.shape == torch.Size((2, 256, 100))
