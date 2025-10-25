import pytest
import torch

from ..models.vae.loss import VAELoss
from ..models.vae.model import VAE

IMAGE_SIZE = 28
IMAGE_SHAPE = (1, 1, IMAGE_SIZE, IMAGE_SIZE)
LATENT_DIM = 64


class TestModel:
    def test_forward_shape_eval(self):
        x = torch.rand(IMAGE_SHAPE)

        model = VAE(image_size=IMAGE_SIZE**2, latent_dim=LATENT_DIM)
        model.eval()

        y = model(x)

        assert y.shape == torch.Size(IMAGE_SHAPE)

    def test_forward_shape_train(self):
        x = torch.rand(IMAGE_SHAPE)

        model = VAE(image_size=IMAGE_SIZE**2, latent_dim=LATENT_DIM)
        model.train()

        y, mu, logsigma = model(x)

        assert y.shape == torch.Size(IMAGE_SHAPE)
        assert mu.shape == torch.Size((1, LATENT_DIM))
        assert logsigma.shape == torch.Size((1, LATENT_DIM))


class TestLoss:
    def setUp(self):
        self.x = torch.rand(IMAGE_SHAPE, requires_grad=True)
        self.y = torch.rand(IMAGE_SHAPE)
        self.mu = torch.rand((1, LATENT_DIM))
        self.logvar = torch.rand((1, LATENT_DIM))

    @pytest.mark.parametrize("loss_type", ["mse", "bce"])
    def test_forward(self, loss_type):
        self.setUp()

        criterion = VAELoss(loss_type=loss_type)

        loss = criterion.forward(self.x, self.mu, self.logvar, self.y)

        assert not torch.isnan(loss)

    @pytest.mark.parametrize("loss_type", ["mse", "bce"])
    def test_backward(self, loss_type):
        self.setUp()

        criterion = VAELoss(loss_type=loss_type)

        loss = criterion.forward(self.x, self.mu, self.logvar, self.y)
        loss.backward()
