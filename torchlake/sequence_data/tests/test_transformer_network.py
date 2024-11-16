import pytest
import torch
from torchlake.common.utils.numerical import causal_mask

from ..models.transformer.network import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    SingleHeadAttention,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)

BATCH_SIZE = 4
SEQ_LEN = 16
INPUT_VOCAB_SIZE = 10
OUTPUT_VOCAB_SIZE = 15
HIDDEN_DIM = 16


class TestScaledDotProductAttention:
    @pytest.mark.parametrize("mask", [None, causal_mask(1, SEQ_LEN, SEQ_LEN)])
    def test_forward_shape(self, mask: torch.Tensor):
        layer = ScaledDotProductAttention()
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        y = layer(x, x, x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestSingleHeadAttention:
    def test_forward_shape(self):
        layer = SingleHeadAttention(HIDDEN_DIM, HIDDEN_DIM)
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        y = layer(x, x, x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestMultiHeadAttention:
    @pytest.mark.parametrize("num_heads", [1, 2, 4])
    def test_forward_shape(self, num_heads: int):
        layer = MultiHeadAttention(HIDDEN_DIM, num_heads)
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        y = layer(x, x, x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestTransformerEncoderBlock:
    def test_forward_shape(self):
        layer = TransformerEncoderBlock(HIDDEN_DIM, 4)
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        y = layer(x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestTransformerDecoderBlock:
    @pytest.mark.parametrize("mask", [None, causal_mask(1, SEQ_LEN, SEQ_LEN)])
    def test_forward_shape(self, mask: torch.Tensor):
        layer = TransformerEncoderBlock(HIDDEN_DIM, 4)
        layer2 = TransformerDecoderBlock(HIDDEN_DIM, 4)
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        encoded = layer(x)
        y = layer2(x, encoded, mask)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))
