from math import ceil

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from ..models.repvgg.model import RepVGG
from ..models.repvgg.network import InceptionModule, RepVGGBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7
NUM_BRANCH = 3
KEYS = [
    "a0",
    "a1",
    "a2",
    "b0",
    "b1",
    "b1g2",
    "b1g4",
    "b2",
    "b2g2",
    "b2g4",
    "b3",
    "b3g2",
    "b3g4",
    "d2se",
]


class TestNetwork:
    @pytest.mark.parametrize(
        "input_channel,output_channel,groups",
        [
            [16, 64, 1],
            [16, 16, 1],
            [16, 64, 16],
            [16, 16, 16],
        ],
    )
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_inception_module_forward_shape(
        self,
        input_channel: int,
        output_channel: int,
        groups: int,
        kernel: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = InceptionModule(
            input_channel,
            output_channel,
            kernel,
            groups=groups,
            num_branch=NUM_BRANCH,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,output_channel,kernel,stride,groups",
        [
            ["depthwise_downsample", 16, 16, 3, 2, 16],
            ["depthwise", 16, 16, 3, 1, 16],
            ["pointwise", 16, 64, 1, 1, 1],
        ],
    )
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_repvgg_block_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        kernel: int,
        stride: int,
        groups: int,
        enable_se: bool,
    ):
        OUTPUT_SHAPE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = RepVGGBlock(
            input_channel,
            output_channel,
            kernel,
            stride,
            groups=groups,
            num_branch=NUM_BRANCH,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SHAPE, OUTPUT_SHAPE)
        )


class TestModel:
    @pytest.mark.parametrize("key", KEYS)
    def test_repvgg_forward_shape(self, key: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = RepVGG(INPUT_CHANNEL, OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))


class TestReparameterize:
    @pytest.mark.parametrize(
        "input_channel,output_channel,stride,groups",
        [
            [16, 64, 1, 1],
            [16, 16, 1, 1],
            [16, 16, 2, 1],
            [16, 64, 1, 16],
            [16, 16, 1, 16],
            [16, 16, 2, 16],
        ],
    )
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_inception_module(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
        groups: int,
        kernel: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = InceptionModule(
            input_channel,
            output_channel,
            kernel,
            stride,
            groups=groups,
            num_branch=NUM_BRANCH,
        ).eval()
        dest = model.reparameterize()

        assert_close(model(x), dest(x))

    @pytest.mark.parametrize("key", ["a0", "b1g4", "d2se"])
    def test_repvgg(self, key: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = RepVGG(INPUT_CHANNEL, OUTPUT_SIZE, key=key).eval()
        dest = model.reparameterize()

        assert_close(model(x), dest(x))

    def test_repvgg_no_bn_after_reparameterize(self):
        model = RepVGG(INPUT_CHANNEL, OUTPUT_SIZE).eval()
        dest = model.reparameterize()

        assert not any(isinstance(m, nn.BatchNorm2d) for m in dest.modules())
