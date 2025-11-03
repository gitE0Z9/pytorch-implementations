import pytest
import torch

from ..models.dcgan.model import DCGANGenerator
from ..models.ebgan.loss import EBGANDiscriminatorLoss, EBGANGeneratorLoss
from ..models.ebgan.model import EBGANDiscriminator

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 32
LATENT_DIM = 16
HIDDEN_DIM = 64


class TestModel:
    @pytest.mark.parametrize("output_latent", (True, False))
    def test_ebgan_discriminator_forward_shape(self, output_latent: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = EBGANDiscriminator(INPUT_CHANNEL, HIDDEN_DIM)

        y = model(x, output_latent=output_latent)

        if output_latent:
            y, z = y
            s = IMAGE_SIZE // (2**model.num_layer)
            assert z.shape == torch.Size(
                (BATCH_SIZE, HIDDEN_DIM * (2 ** (model.num_layer - 1)) * s * s)
            )

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )


class TestLoss:
    @pytest.mark.parametrize("lambda_pt", (0, 1e-1))
    def test_ebgan_generator_loss_forward(self, lambda_pt: float):
        z = torch.rand((BATCH_SIZE, LATENT_DIM))

        g = DCGANGenerator(LATENT_DIM, INPUT_CHANNEL)
        d = EBGANDiscriminator(INPUT_CHANNEL, HIDDEN_DIM)
        xhat = g(z)

        if lambda_pt > 0:
            yhat_xhat, z_xhat = d(xhat, output_latent=True)
        else:
            yhat_xhat = d(xhat)

        criterion = EBGANGeneratorLoss(lambda_pt)
        if lambda_pt > 0:
            loss = criterion(yhat_xhat, xhat, z_xhat)
        else:
            loss = criterion(yhat_xhat, xhat)

        assert not torch.isnan(loss)

    def test_ebgan_discriminator_loss_forward(self):
        z = torch.rand((BATCH_SIZE, LATENT_DIM))
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        g = DCGANGenerator(LATENT_DIM, INPUT_CHANNEL)
        d = EBGANDiscriminator(INPUT_CHANNEL, HIDDEN_DIM)
        xhat = g(z)
        yhat_xhat = d(xhat)
        yhat_x = d(x)

        criterion = EBGANDiscriminatorLoss()
        loss = criterion(yhat_x, yhat_xhat, x, xhat)

        assert not torch.isnan(loss)
