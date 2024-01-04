import torch

from ..models.seq2seq.network import Seq2SeqEncoder, Seq2SeqDecoder
from ..models.seq2seq.model import Seq2Seq


def test_forward_shape_encoder():
    x = torch.randint(0, 100, (2, 256))
    model = Seq2SeqEncoder(100, 16, 16)
    ht, ct = model(x)

    assert ht.shape == torch.Size((1, 2, 16))
    assert ct.shape == torch.Size((1, 2, 16))


def test_forward_shape_decoder():
    x = torch.randint(0, 100, (2, 256))
    h, c = torch.rand((1, 2, 16)), torch.rand((1, 2, 16))
    model = Seq2SeqDecoder(100, 16, 16, 100)
    y = model(x, h, c)

    assert y.shape == torch.Size((2, 256, 100))


def test_forward_shape_model():
    x = torch.randint(0, 100, (2, 256))
    y = torch.randint(0, 100, (2, 256))
    encoder = Seq2SeqEncoder(100, 16, 16)
    decoder = Seq2SeqDecoder(100, 16, 16, 100)
    model = Seq2Seq(encoder, decoder)
    output = model(x, y)

    assert output.shape == torch.Size((2, 256, 100))
