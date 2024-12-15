import pytest
import torch
from torchlake.common.models import ViTFeatureExtractor

from ..models.setr.model import SETR
from ..models.setr.network import MLADecoder, PUPDecoder


class TestPUP:
    def test_forward_shape(self):
        x = torch.rand((1, 196, 8))

        model = PUPDecoder(8, 21)

        y = model(x)

        assert y.shape == torch.Size((1, 21, 224, 224))


class TestMLA:
    def test_forward_shape(self):
        x = [torch.rand((1, 196, 8)) for _ in range(4)]

        model = MLADecoder(8, 21)

        y = model(x)

        assert y.shape == torch.Size((1, 21, 224, 224))


class TestModel:
    @pytest.mark.parametrize(
        "decoder,target_layer_names",
        [("PUP", (11,)), ("MLA", (2, 5, 8, 11))],
    )
    def test_forward_shape(self, decoder: str, target_layer_names: tuple[int]):
        x = torch.rand((1, 3, 224, 224))

        backbone = ViTFeatureExtractor("b16", trainable=False)
        backbone.fix_target_layers(target_layer_names)
        model = SETR(backbone, 21, decoder)

        y = model(x)

        assert y.shape == torch.Size((1, 21, 224, 224))
