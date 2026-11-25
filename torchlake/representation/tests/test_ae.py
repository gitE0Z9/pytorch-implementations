import pytest
import torch
from torch import nn

from ..models.ae.model import AutoEncoder

BATCH_SIZE = 2
INPUT_CHANNEL = 1
IMAGE_SIZE = 28
LATENT_DIM = 16


class TestModel:
    @pytest.mark.parametrize("output_latent", (True, False))
    def test_autoencoder_forward_shape(self, output_latent: bool):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        m = AutoEncoder(
            INPUT_CHANNEL * IMAGE_SIZE * IMAGE_SIZE,
            LATENT_DIM,
            activation=nn.Sigmoid(),
        )
        y = m(x, output_latent=output_latent)

        if output_latent:
            y, z = y
            assert z.shape == torch.Size((BATCH_SIZE, LATENT_DIM))

        assert y.shape == x.shape

    @pytest.mark.parametrize("num_stacked_layer", (2, 3, 4))
    @pytest.mark.parametrize("output_latent", (True, False))
    def test_stacked_autoencoder_forward_shape(
        self,
        num_stacked_layer: int,
        output_latent: bool,
    ):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        m = AutoEncoder(
            INPUT_CHANNEL * IMAGE_SIZE * IMAGE_SIZE,
            LATENT_DIM,
            activation=nn.Sigmoid(),
        )
        for _ in range(num_stacked_layer - 1):
            m.add_stack()

        y = m(x, output_latent=output_latent)

        if output_latent:
            y, z = y
            assert z.shape == (BATCH_SIZE, LATENT_DIM >> (num_stacked_layer - 1))

        assert y.shape == x.shape
