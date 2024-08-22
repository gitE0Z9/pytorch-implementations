import pytest
import torch

from ..models.bam.model import BamResNet


class TestBamResNet:
    @pytest.mark.parametrize("key", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resnet_forward_shape(self, key: int, pre_activation: bool):
        x = torch.randn(2, 3, 224, 224)
        model = BamResNet(
            output_size=5,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))
