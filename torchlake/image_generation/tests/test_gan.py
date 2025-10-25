import torch

from torchlake.common.utils.random import generate_normal
from ..models.gan import (
    GANDiscriminator,
    GANGenerator,
    GANDiscriminatorLoss,
    GANGeneratorLoss,
)

BATCH_SIZE = 1
INPUT_CHANNEL = 3
LATENT_DIM = 128
HIDDEN_DIM = 256
IMAGE_SIZE = 32


class TestModel:
    def test_generator_forward_shape(self):
        z = generate_normal(BATCH_SIZE, IMAGE_SIZE)

        model = GANGenerator(
            LATENT_DIM,
            HIDDEN_DIM,
            (INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE),
        )

        y = model(z)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )

    def test_discriminator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = GANDiscriminator(HIDDEN_DIM, (INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, 1))


class TestLoss:
    def test_generator_loss_forward(self):
        z = generate_normal(BATCH_SIZE, LATENT_DIM)

        g = GANGenerator(
            LATENT_DIM,
            HIDDEN_DIM,
            (INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE),
        )
        d = GANDiscriminator(HIDDEN_DIM, (INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        yhat_xhat = d(g(z))

        criterion = GANGeneratorLoss()
        loss = criterion(yhat_xhat)

        assert not torch.isnan(loss)

    def test_discriminator_loss_forward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        z = generate_normal(BATCH_SIZE, LATENT_DIM)

        g = GANGenerator(
            LATENT_DIM,
            HIDDEN_DIM,
            (INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE),
        )
        d = GANDiscriminator(HIDDEN_DIM, (INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        yhat_x = d(x)
        yhat_xhat = d(g(z))

        criterion = GANDiscriminatorLoss()
        loss = criterion(yhat_x, yhat_xhat)

        assert not torch.isnan(loss)
