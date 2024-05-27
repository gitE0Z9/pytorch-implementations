import torch
from ..models import DaNet
from ..models.dual_attention.network import (
    DualAttention2d,
    SpatialAttention2d,
    ChannelAttention2d,
)


class TestDaNet:
    def test_training_forward_shape(self):
        x = torch.rand((16, 3, 224, 224))

        model = DaNet(2048, 21)

        y = model(x)

        assert y.shape == torch.Size((16, 21, 224, 224))
        # assert aux.shape == torch.Size((16, 21, 224, 224))

    def test_eval_forward_shape(self):
        x = torch.rand((16, 3, 224, 224))

        model = DaNet(2048, 21).eval()

        y = model(x)

        assert y.shape == torch.Size((16, 21, 224, 224))


class TestDualAttention2d:
    def test_sa_forward_shape(self):
        x = torch.rand((16, 2048, 7, 7))

        model = SpatialAttention2d(2048)

        y = model(x)

        assert y.shape == torch.Size((16, 2048, 7, 7))

    def test_ca_forward_shape(self):
        x = torch.rand((16, 2048, 7, 7))

        model = ChannelAttention2d()

        y = model(x)

        assert y.shape == torch.Size((16, 2048, 7, 7))

    def test_forward_shape(self):
        x = torch.rand((16, 2048, 7, 7))

        model = DualAttention2d(2048)

        y = model(x)

        assert y.shape == torch.Size((16, 2048, 7, 7))
