import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.testing import assert_close

from torchlake.common.utils.numerical import causal_mask

from ..models.transformer.model import (
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
)
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


class TestNetwork:
    @pytest.mark.parametrize("mask", [None, causal_mask(1, SEQ_LEN, SEQ_LEN)])
    def test_scaled_dot_product_attention_equalness(self, mask: torch.Tensor):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        model = ScaledDotProductAttention()

        yhat = model(x, x, x, mask)
        if mask is not None:
            mask = mask.bool()
        y = scaled_dot_product_attention(x, x, x, attn_mask=mask)

        assert yhat.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))
        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))
        assert_close(yhat, y)

    def test_single_head_attention_forward_shape(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        model = SingleHeadAttention(HIDDEN_DIM, HIDDEN_DIM)

        y = model(x, x, x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))

    @pytest.mark.parametrize("num_heads", [1, 2, 4])
    def test_multi_head_attention_forward_shape(self, num_heads: int):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        model = MultiHeadAttention(HIDDEN_DIM, num_heads)

        y = model(x, x, x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))

    def test_transformer_encoder_block_forward_shape(self):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        model = TransformerEncoderBlock(HIDDEN_DIM, 4)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))

    @pytest.mark.parametrize("mask", [None, causal_mask(1, SEQ_LEN, SEQ_LEN)])
    def test_transformer_decoder_block_forward_shape(self, mask: torch.Tensor):
        layer = TransformerEncoderBlock(HIDDEN_DIM, 4)
        layer2 = TransformerDecoderBlock(HIDDEN_DIM, 4)
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        encoded = layer(x)
        y = layer2(x, encoded, mask)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))


class TestModel:
    def test_transformer_encoder_forward_shape(self):
        x = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = TransformerEncoder(INPUT_VOCAB_SIZE, hidden_dim=HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM))

    @pytest.mark.parametrize("output_length", [SEQ_LEN, 2 * SEQ_LEN])
    @pytest.mark.parametrize("causal_mask", [True, False])
    def test_transformer_decoder_forward_shape(
        self,
        output_length: int,
        causal_mask: bool,
    ):
        x = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, output_length))
        y = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
        model = TransformerDecoder(
            OUTPUT_VOCAB_SIZE,
            OUTPUT_VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
            causal_mask=causal_mask,
        )

        z = model(x, y)

        assert z.shape == torch.Size((BATCH_SIZE, output_length, OUTPUT_VOCAB_SIZE))

    def test_transformer_forward_shape(self):
        x = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        encoder = TransformerEncoder(INPUT_VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
        decoder = TransformerDecoder(
            OUTPUT_VOCAB_SIZE,
            OUTPUT_VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
        )
        model = Transformer(encoder, decoder)

        z = model.forward(x, y)

        assert z.shape == torch.Size((BATCH_SIZE, SEQ_LEN, OUTPUT_VOCAB_SIZE))
