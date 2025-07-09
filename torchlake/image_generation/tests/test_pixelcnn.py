import pytest
import torch
from torch.testing import assert_close

from ..models.pixelcnn.model import PixelCNN
from ..models.pixelcnn.network import BottleNeck, MaskedConv2d

BATCH_SIZE = 2
IMAGE_SIZE = 32
HIDDEN_DIM = 8
OUTPUT_SIZE = 5


class TestNetwork:
    @pytest.mark.parametrize("mask_type", ["A", "B"])
    def test_masked_conv_2d_build_mask(self, mask_type: str):
        k = 3
        model = MaskedConv2d(
            3,
            3,
            k,
            padding=1,
            mask_type=mask_type,
            mask_groups=3,
        )

        if mask_type == "A":
            expected = torch.ones(3, 3).tril_() - torch.eye(3)
        else:
            expected = torch.ones(3, 3).tril_()

        assert_close(model.mask[:, :, k // 2, k // 2], expected)

    @pytest.mark.parametrize("mask_type", ["A", "B"])
    @pytest.mark.parametrize("mask_groups", [1, 3])
    def test_masked_conv_2d_forward_shape(self, mask_type: str, mask_groups: int):
        x = torch.rand(BATCH_SIZE, mask_groups * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)

        model = MaskedConv2d(
            mask_groups * HIDDEN_DIM,
            mask_groups * HIDDEN_DIM,
            3,
            padding=1,
            mask_type=mask_type,
            mask_groups=mask_groups,
        )

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, mask_groups * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        )

    @pytest.mark.parametrize("mask_groups", [1, 3])
    def test_bottleneck_forward_shape(self, mask_groups: int):
        x = torch.rand(BATCH_SIZE, 2 * HIDDEN_DIM * mask_groups, IMAGE_SIZE, IMAGE_SIZE)

        model = BottleNeck(HIDDEN_DIM * mask_groups, mask_groups=mask_groups)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 2 * HIDDEN_DIM * mask_groups, IMAGE_SIZE, IMAGE_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("in_c", [1, 3])
    def test_forward_shape(self, in_c: int):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = PixelCNN(in_c, 256, HIDDEN_DIM, num_layer=6)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, in_c, 256, IMAGE_SIZE, IMAGE_SIZE))
