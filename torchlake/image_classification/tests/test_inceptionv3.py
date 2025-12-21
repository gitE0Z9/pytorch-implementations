from typing import Sequence
import pytest
import torch

from ..models.inceptionv3.model import InceptionV3
from ..models.inceptionv3.network import (
    InceptionBlockV3,
    AuxiliaryClassifierV3,
    AsymmetricConv2d,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 299
NUM_CLASS = 10


class TestNetwork:
    @pytest.mark.parametrize("mode", ("parallel", "sequential"))
    @pytest.mark.parametrize("kernel_first", (True, False))
    def test_asymmetric_conv2d_forward_shape(self, mode: str, kernel_first: bool):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        m = AsymmetricConv2d(
            HIDDEN_DIM,
            (HIDDEN_DIM, HIDDEN_DIM),
            kernel=7,
            mode=mode,
            kernel_first=kernel_first,
        )

        y = m(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                HIDDEN_DIM if mode == "sequential" else 2 * HIDDEN_DIM,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

    @pytest.mark.parametrize(
        "output_channels,kernels,mode",
        (
            (
                (
                    HIDDEN_DIM,
                    (48, HIDDEN_DIM),
                    (64, 96, HIDDEN_DIM),
                    HIDDEN_DIM,
                ),
                (1, (1, 5), (1, 3, 3)),
                "sequential",
            ),
            (
                (
                    HIDDEN_DIM,
                    (HIDDEN_DIM, (HIDDEN_DIM, HIDDEN_DIM)),
                    (HIDDEN_DIM, (128, 128), (HIDDEN_DIM, HIDDEN_DIM)),
                    HIDDEN_DIM,
                ),
                (1, (1, (7, False)), (1, (7, True), (7, True))),
                "sequential",
            ),
            (
                (
                    HIDDEN_DIM,
                    (HIDDEN_DIM, (HIDDEN_DIM, HIDDEN_DIM)),
                    (HIDDEN_DIM, 384, (HIDDEN_DIM, HIDDEN_DIM)),
                    HIDDEN_DIM,
                ),
                (1, (1, (3, True)), (1, 3, (3, False))),
                "parallel",
            ),
        ),
    )
    def test_inception_block_v3_forward_shape(
        self,
        output_channels: Sequence[int],
        kernels: Sequence[int],
        mode: str,
    ):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionBlockV3(
            HIDDEN_DIM,
            output_channels,
            kernels=kernels,
            mode=mode,
        )

        y = m(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                4 * HIDDEN_DIM if mode == "sequential" else 6 * HIDDEN_DIM,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

    def test_auxiliary_classifier_v3_forward_shape(self):
        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        m = AuxiliaryClassifierV3(
            HIDDEN_DIM,
            NUM_CLASS,
            (HIDDEN_DIM, HIDDEN_DIM),
            kernel=5,
        )

        y = m(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS))


class TestModel:
    @pytest.mark.parametrize("is_training", (True, False))
    def test_inception_bn_forward_shape(self, is_training: bool):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = InceptionV3(INPUT_CHANNEL, NUM_CLASS)

        if is_training:
            m.train()
        else:
            m.eval()

        y = m(x)

        if is_training:
            for ele in y:
                assert ele.shape == torch.Size((BATCH_SIZE, NUM_CLASS))
        else:
            assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS))
