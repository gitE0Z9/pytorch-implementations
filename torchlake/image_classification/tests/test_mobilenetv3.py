import pytest
import torch

from ..models.mobilenetv3.model import MobileNetV3
from ..models.mobilenetv3.network import (
    DepthwiseSeparableConv2dV3,
    LinearBottleneckV3,
    InvertedResidualBlockV3,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
OUTPUT_SIZE = 5
IMAGE_SIZE = 224
HIDDEN_DIM = 64
FEATURE_MAP_SIZE = 7


class TestNetwork:
    @pytest.mark.parametrize(
        "name,enable_relu,enable_se",
        [
            ["enable_relu", True, False],
            ["enable_relu", False, False],
            ["enable_se", False, False],
            ["enable_se", False, True],
        ],
    )
    def test_linear_bottleneck_v3_forward_shape(
        self,
        name: str,
        enable_relu: bool,
        enable_se: bool,
    ):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = LinearBottleneckV3(
            HIDDEN_DIM,
            HIDDEN_DIM + 2,
            3,
            expansion_size=24,
            enable_relu=enable_relu,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "name,enable_relu,enable_se",
        [
            ["enable_relu", True, False],
            ["enable_relu", False, False],
            ["enable_se", False, False],
            ["enable_se", False, True],
        ],
    )
    def test_inverted_residual_block_v3_forward_shape(
        self,
        name: str,
        enable_relu: bool,
        enable_se: bool,
    ):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = InvertedResidualBlockV3(
            HIDDEN_DIM,
            HIDDEN_DIM + 2,
            3,
            expansion_size=24,
            enable_relu=enable_relu,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_depthwise_separable_conv2d_v3_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = DepthwiseSeparableConv2dV3(HIDDEN_DIM, HIDDEN_DIM + 2)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize(
        "name,key",
        [
            ["large", "large"],
            ["small", "small"],
        ],
    )
    def test_mobilenetv3_forward_shape(self, name: str, key: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = MobileNetV3(output_size=OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
