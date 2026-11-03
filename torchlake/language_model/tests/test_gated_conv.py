import pytest
import torch

from torchlake.common.schemas.nlp import NLPContext

from ..models.gate_conv.model import GatedConvLM
from ..models.gate_conv.network import Bottleneck, GatedCausalConv1d, ResBlock

BATCH_SIZE = 4
VOCAB_SIZE = 50
EMBED_DIM = 4
HIDDEN_DIM = 8
SEQ_LEN = 256
CONTEXT = NLPContext(device="cpu")


class TestNetwork:
    def test_gated_causal_conv1d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN))
        model = GatedCausalConv1d(HIDDEN_DIM, HIDDEN_DIM + 2, 3, padding=2)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM + 2, SEQ_LEN))

    def test_bottleneck_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN))
        model = Bottleneck(
            HIDDEN_DIM,
            (HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM * 2),
            (1, 5, 1),
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM * 2, SEQ_LEN))

    def test_resblock_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN))
        model = ResBlock(
            HIDDEN_DIM,
            HIDDEN_DIM * 2,
            block=Bottleneck(
                HIDDEN_DIM,
                (HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM * 2),
                (1, 5, 1),
            ),
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM * 2, SEQ_LEN))


class TestModel:
    @pytest.mark.parametrize("key", ("8", "9", "13", "14", "8B", "14B"))
    @pytest.mark.parametrize("head_type", ("conv", "linear"))
    def test_gated_conv_lm_forward_shape(self, key: str, head_type: str):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        model = GatedConvLM(
            VOCAB_SIZE,
            EMBED_DIM,
            key=key,
            head_type=head_type,
            context=CONTEXT,
        )

        y = model(x)

        if head_type == "conv":
            assert y.shape == torch.Size((BATCH_SIZE, VOCAB_SIZE, SEQ_LEN))
        else:
            assert y.shape == torch.Size((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))

    @pytest.mark.parametrize("head_type", ("conv", "linear"))
    def test_gated_conv_lm_output_embedding(self, head_type: str):
        model = GatedConvLM(
            VOCAB_SIZE,
            EMBED_DIM,
            key="8",
            head_type=head_type,
            context=CONTEXT,
        )

        y = model.output_embedding

        assert y.weight.shape == torch.Size((VOCAB_SIZE, model.feature_dim))
