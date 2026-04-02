import torch
from torchlake.super_resolution.models.vdsr import VDSR

BATCH_SIZE = 4
INPUT_CHANNEL = 3
IMAGE_SIZE = 256

class TestModel:
    def test_vdsr_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        model = VDSR(INPUT_CHANNEL)

        y = model(x)

        assert y.shape == torch.Size(
             (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )