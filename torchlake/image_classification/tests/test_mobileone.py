import pytest
import torch
from torch import nn
from torch.testing import assert_close

from ..models.mobileone.model import MobileOne

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5


class TestModel:
    @pytest.mark.parametrize("key", ["s0", "s1", "s2", "s3", "s4", "mu0", "mu1", "mu2"])
    def test_mobileone_forward_shape(self, key: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = MobileOne(INPUT_CHANNEL, OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))


class TestReparameterize:
    @pytest.mark.parametrize("key", ["s0", "s4", "mu0"])
    def test_mobileone(self, key: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = MobileOne(INPUT_CHANNEL, OUTPUT_SIZE, key=key).eval()
        dest = model.reparameterize()

        # activations of an untrained deep model are large, so compare relatively
        assert_close(model(x), dest(x), rtol=1e-4, atol=1e-3)

    def test_mobileone_no_bn_after_reparameterize(self):
        model = MobileOne(INPUT_CHANNEL, OUTPUT_SIZE).eval()
        dest = model.reparameterize()

        assert not any(isinstance(m, nn.BatchNorm2d) for m in dest.modules())
