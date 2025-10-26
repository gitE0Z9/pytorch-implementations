import torch
import torch.nn.functional as F

from torchlake.common.utils.random import generate_normal

from ..models.acgan import (
    ACGANDiscriminator,
    ACGANDiscriminatorLoss,
    ACGANGeneratorLoss,
)
from ..models.dcgan import DCGANGenerator

BATCH_SIZE = 2
INPUT_CHANNEL = 3
NUM_CLASS = 10
LATENT_DIM = 16
HIDDEN_DIM = 64
IMAGE_SIZE = 32


class TestModel:
    def test_acgan_discriminator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        # 32 -> 1 => downscale (1 + #block) times
        # 64 -> 1024
        model = ACGANDiscriminator(
            INPUT_CHANNEL,
            NUM_CLASS,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
            num_block=4,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, 1 + NUM_CLASS))


class TestLoss:
    def test_acgan_discriminator_loss_forward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE,))
        z = generate_normal(BATCH_SIZE, LATENT_DIM)
        c = F.one_hot(y, NUM_CLASS)

        g = DCGANGenerator(
            LATENT_DIM + NUM_CLASS,
            INPUT_CHANNEL,
            hidden_dim=HIDDEN_DIM,
            init_shape=(2, 2),
        )
        d = ACGANDiscriminator(
            INPUT_CHANNEL,
            NUM_CLASS,
            HIDDEN_DIM,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        yhat_x = d(x)
        yhat_xhat = d(g(torch.cat((z, c), 1)))

        criterion = ACGANDiscriminatorLoss()
        loss = criterion(yhat_x, yhat_xhat, y)

        assert not torch.isnan(loss)

    def test_acgan_generator_loss_forward(self):
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE,))
        z = generate_normal(BATCH_SIZE, LATENT_DIM)
        c = F.one_hot(y, NUM_CLASS)

        g = DCGANGenerator(
            LATENT_DIM + NUM_CLASS,
            INPUT_CHANNEL,
            hidden_dim=HIDDEN_DIM,
            init_shape=(2, 2),
        )
        d = ACGANDiscriminator(
            INPUT_CHANNEL,
            NUM_CLASS,
            HIDDEN_DIM,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        yhat_xhat = d(g(torch.cat((z, c), 1)))

        criterion = ACGANGeneratorLoss()
        loss = criterion(yhat_xhat, y)

        assert not torch.isnan(loss)
