from pathlib import Path
import pytest
import torch
import random
from torch.testing import assert_close
from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.darknet19 import DarkNet19FeatureExtractor

from torchlake.object_detection.utils.train import build_flatten_targets

from ..constants.schema import DetectorContext
from ..models.yolov2.model import YOLOV2
from ..models.yolov2.loss import YOLOV2Loss, YOLO9000Loss


BATCH_SIZE = 2
GRID_SIZE = 13
MAX_OBJECT_SIZE = 10
NUM_CLASS = 20


CONTEXT = DetectorContext(
    detector_name="yolov2",
    dataset="voc",
    device="cpu",
    num_classes=NUM_CLASS,
    num_anchors=5,
    anchors_path=__file__.replace("test_yolov2.py", "fake.anchors.txt"),
)

OUTPUT_SIZE = (CONTEXT.num_classes + 5) * CONTEXT.num_anchors


class TestYOLOV2:
    def test_output_shape(self):
        backbone = DarkNet19FeatureExtractor("last_conv", trainable=False)
        backbone.fix_target_layers(["3_1", "4_1"])

        model = YOLOV2(
            backbone,
            CONTEXT,
            backbone_feature_dims=backbone.feature_dims[-2:],
        )
        test_x = torch.rand(2, 3, 416, 416)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, OUTPUT_SIZE, 13, 13]))


class TestYOLOV2ResNet:
    @pytest.mark.parametrize("network_name", ["resnet18", "resnet34", "resnet50"])
    def test_output_shape(self, network_name: str):
        backbone = ResNetFeatureExtractor(network_name, "block", False)
        backbone.fix_target_layers(["3_1", "4_1"])

        model = YOLOV2(
            backbone,
            CONTEXT,
            backbone_feature_dims=backbone.feature_dims[-2:],
        )
        test_x = torch.rand(2, 3, 416, 416)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, OUTPUT_SIZE, 13, 13]))


class TestYOLOv2Loss:
    def setUp(self):
        self.x = torch.rand(
            BATCH_SIZE,
            OUTPUT_SIZE,
            GRID_SIZE,
            GRID_SIZE,
        )  # .requires_grad_()
        self.y = [
            [
                [
                    random.random(),
                    random.random(),
                    random.randrange(0, GRID_SIZE),
                    random.randrange(0, GRID_SIZE),
                    random.randrange(0, NUM_CLASS),
                ]
                for _ in range(random.randint(1, MAX_OBJECT_SIZE))
            ]
            for _ in range(BATCH_SIZE)
        ]

    def test_iou_box(self):
        self.setUp()

        criterion = YOLOV2Loss(CONTEXT, iou_threshold=0)
        y, span = build_flatten_targets(self.y, (GRID_SIZE, GRID_SIZE))

        targets = criterion.match(y, span, GRID_SIZE, GRID_SIZE)
        # assert (iou - labels[:, :, 4:5, :, :]).sum() < 1e-2, "iou is too far away"
        assert_close(
            targets.shape,
            torch.Size([BATCH_SIZE, CONTEXT.num_anchors * GRID_SIZE * GRID_SIZE, 7]),
        )

    def test_forward(self):
        self.setUp()

        criterion = YOLOV2Loss(CONTEXT)

        loss = criterion(self.x, self.y)
        assert not torch.isnan(loss)

    def test_backward(self):
        self.setUp()

        criterion = YOLOV2Loss(CONTEXT)

        loss = criterion(self.x, self.y)
        loss.backward()
