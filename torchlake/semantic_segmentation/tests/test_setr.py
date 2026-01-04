import pytest
import torch
from torchlake.common.models import ViTFeatureExtractor

from ..models.setr.model import SETR
from ..models.setr.network import MLADecoder, PUPDecoder

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
NUM_CLASS = 21
IMAGE_SIZE = 224
NUM_PATCH = 14**2


class TestNetwork:
    def test_pup_decoder_forward_shape(self):
        x = [torch.rand((BATCH_SIZE, NUM_PATCH, HIDDEN_DIM))]

        model = PUPDecoder(HIDDEN_DIM, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_mla_decoder_forward_shape(self):
        x = [torch.rand((BATCH_SIZE, NUM_PATCH, HIDDEN_DIM)) for _ in range(4)]

        model = MLADecoder(HIDDEN_DIM, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))


class TestModel:
    @pytest.mark.parametrize("decoder", ["PUP", "MLA"])
    def test_forward_shape(self, decoder: str):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ViTFeatureExtractor("b16", trainable=False)
        model = SETR(backbone, NUM_CLASS, decoder)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
