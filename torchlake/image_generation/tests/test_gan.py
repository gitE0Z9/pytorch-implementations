import torch

from torchlake.common.utils.random import generate_normal
from ..models.gan import GANDiscriminator, GANGenerator


class TestModel:
    def test_generator_forward_shape(self):
        z = generate_normal(1, 128)

        model = GANGenerator(128, 256, (3, 32, 32))

        y = model(z)

        assert y.shape == torch.Size((1, 3, 32, 32))

    def test_discriminator_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = GANDiscriminator(256, (3, 32, 32))

        y = model(x)

        assert y.shape == torch.Size((1, 1))
