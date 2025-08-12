import random

import pytest
import torch
from torch.testing import assert_close

from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.darknet53 import DarkNet53FeatureExtractor
from torchlake.object_detection.utils.train import build_flatten_targets

from ..constants.schema import DetectorContext
from ..models.yolov3.loss import YOLOV3Loss
from ..models.yolov3.model import YOLOV3

BATCH_SIZE = 2
IMAGE_SIZE = 416
GRID_SIZES = (IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8)
MAX_OBJECT_SIZE = 10
NUM_CLASS = 20


CONTEXT = DetectorContext(
    detector_name="yolov3",
    dataset="voc",
    device="cpu",
    num_classes=NUM_CLASS,
    num_anchors=(3, 3, 3),
    anchors_path="",
)

OUTPUT_SIZES = (
    (CONTEXT.num_classes + 5) * num_anchor for num_anchor in CONTEXT.num_anchors
)


class TestYOLOV3:
    def test_output_shape(self):
        backbone = DarkNet53FeatureExtractor("block", trainable=False)
        backbone.fix_target_layers(["2_1", "3_1", "4_1"])

        model = YOLOV3(
            backbone,
            CONTEXT,
            hidden_dim_8x=backbone.feature_dims[-3],
            hidden_dim_16x=backbone.feature_dims[-2],
            hidden_dim_32x=backbone.feature_dims[-1],
        )
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        outputs: list[torch.Tensor] = model(x)
        for output, OUTPUT_SIZE, GRID_SIZE in zip(outputs, OUTPUT_SIZES, GRID_SIZES):
            assert_close(
                output.shape,
                torch.Size([BATCH_SIZE, OUTPUT_SIZE, GRID_SIZE, GRID_SIZE]),
            )


class TestYOLOV3ResNet:
    @pytest.mark.parametrize("network_name", ["resnet18", "resnet34", "resnet50"])
    def test_output_shape(self, network_name: str):
        backbone = ResNetFeatureExtractor(network_name, "block", trainable=False)
        backbone.fix_target_layers(["2_1", "3_1", "4_1"])

        model = YOLOV3(
            backbone,
            CONTEXT,
            hidden_dim_8x=backbone.feature_dims[-3],
            hidden_dim_16x=backbone.feature_dims[-2],
            hidden_dim_32x=backbone.feature_dims[-1],
        )
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        outputs: list[torch.Tensor] = model(x)
        for output, OUTPUT_SIZE, GRID_SIZE in zip(outputs, OUTPUT_SIZES, GRID_SIZES):
            assert_close(
                output.shape,
                torch.Size([BATCH_SIZE, OUTPUT_SIZE, GRID_SIZE, GRID_SIZE]),
            )
