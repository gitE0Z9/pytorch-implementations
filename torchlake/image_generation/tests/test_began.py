import pytest
import torch
from ..models.began.model import BEGANGenerator, BEGANDiscriminator
from ..models.began.loss import BEGANDiscriminatorLoss

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 32
LATENT_DIM = 16
HIDDEN_DIM = 64


class TestModel:
    def test_began_generator_forward_shape(self):
        x = torch.rand(BATCH_SIZE, LATENT_DIM)
        m = BEGANGenerator(LATENT_DIM, INPUT_CHANNEL, HIDDEN_DIM, (8, 8))

        y = m(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )

    @pytest.mark.parametrize("output_latent", (True, False))
    def test_began_discriminator_forward_shape(self, output_latent: bool):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = BEGANDiscriminator(
            INPUT_CHANNEL,
            LATENT_DIM,
            HIDDEN_DIM,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        y = m(x, output_latent=output_latent)

        if output_latent:
            y, z = y
            assert z.shape == torch.Size((BATCH_SIZE, LATENT_DIM))

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )


class TestLoss:
    def test_began_discriminator_loss_forward(self):
        z = torch.rand((BATCH_SIZE, LATENT_DIM))
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        g = BEGANGenerator(LATENT_DIM, INPUT_CHANNEL, HIDDEN_DIM, (8, 8))
        d = BEGANDiscriminator(
            INPUT_CHANNEL,
            LATENT_DIM,
            HIDDEN_DIM,
            (IMAGE_SIZE, IMAGE_SIZE),
        )
        xhat = g(z)
        yhat_xhat = d(xhat)
        yhat_x = d(x)
        criterion = BEGANDiscriminatorLoss()

        loss = criterion(yhat_x, yhat_xhat, x, xhat)

        assert not torch.isnan(loss)
