import torch

from ..models.pix2pix.loss import Pix2PixGeneratorLoss, Pix2PixDiscriminatorLoss
from ..models.pix2pix.model import Pix2PixDiscriminator, Pix2PixGenerator
from ..models.pix2pix.network import DownSampling, UpSampling

BATCH_SIZE = 2
IMAGE_SIZE = 256
INPUT_CHANNEL = 3
OUTPUT_SIZE = 3
HIDDEN_DIM = 8


class TestNetwork:
    def test_downsampling_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 8, IMAGE_SIZE // 8))
        model = DownSampling(HIDDEN_DIM, HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 16, IMAGE_SIZE // 16)
        )

    def test_upsampling_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM + 2, IMAGE_SIZE // 8, IMAGE_SIZE // 8))
        z = torch.rand((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 4, IMAGE_SIZE // 4))
        model = UpSampling(HIDDEN_DIM + 2, HIDDEN_DIM)

        y = model(x, z)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM * 2, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        )


class TestModel:
    def test_pix2pix_generator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        model = Pix2PixGenerator(INPUT_CHANNEL, INPUT_CHANNEL, HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )

    def test_pix2pix_discriminator_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        z = torch.rand((BATCH_SIZE, OUTPUT_SIZE, IMAGE_SIZE, IMAGE_SIZE))
        model = Pix2PixDiscriminator(INPUT_CHANNEL + OUTPUT_SIZE, HIDDEN_DIM)

        y = model(x, z)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 1, IMAGE_SIZE // 8 - 2, IMAGE_SIZE // 8 - 2)
        )


class TestLoss:
    def test_pix2pix_generator_loss_forward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        w = torch.rand((BATCH_SIZE, OUTPUT_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        generator = Pix2PixGenerator(INPUT_CHANNEL, OUTPUT_SIZE, HIDDEN_DIM)
        discriminator = Pix2PixDiscriminator(INPUT_CHANNEL + OUTPUT_SIZE, HIDDEN_DIM)

        what = generator(x)

        criterion = Pix2PixGeneratorLoss(lambda_coef=0.5)
        loss = criterion(discriminator(what, x), w, what)

        assert not torch.isnan(loss)

    def test_pix2pix_generator_loss_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        w = torch.rand((BATCH_SIZE, OUTPUT_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        generator = Pix2PixGenerator(INPUT_CHANNEL, OUTPUT_SIZE, HIDDEN_DIM)
        discriminator = Pix2PixDiscriminator(INPUT_CHANNEL + OUTPUT_SIZE, HIDDEN_DIM)

        what = generator(x)

        criterion = Pix2PixGeneratorLoss(lambda_coef=0.5)
        loss = criterion(discriminator(what, x), w, what)

        loss.backward()

    def test_pix2pix_discriminator_loss_forward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        w = torch.rand((BATCH_SIZE, OUTPUT_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        generator = Pix2PixGenerator(INPUT_CHANNEL, OUTPUT_SIZE, HIDDEN_DIM)
        discriminator = Pix2PixDiscriminator(INPUT_CHANNEL + OUTPUT_SIZE, HIDDEN_DIM)

        what = generator(x)
        yhat_what = discriminator(what, x)
        yhat_w = discriminator(w, x)

        criterion = Pix2PixDiscriminatorLoss()
        loss = criterion(yhat_what, yhat_w)

        assert not torch.isnan(loss)

    def test_pix2pix_discriminator_loss_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        w = torch.rand((BATCH_SIZE, OUTPUT_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        generator = Pix2PixGenerator(INPUT_CHANNEL, OUTPUT_SIZE, HIDDEN_DIM)
        discriminator = Pix2PixDiscriminator(INPUT_CHANNEL + OUTPUT_SIZE, HIDDEN_DIM)

        what = generator(x)
        yhat_what = discriminator(what, x)
        yhat_w = discriminator(w, x)

        criterion = Pix2PixDiscriminatorLoss()
        loss = criterion(yhat_what, yhat_w)

        loss.backward()
