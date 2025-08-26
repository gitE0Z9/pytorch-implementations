import random

import pytest
import torch
from torch.testing import assert_close

from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.darknet19 import DarkNet19FeatureExtractor
from torchlake.object_detection.utils.train import build_flatten_targets

from ..constants.schema import DetectorContext
from ..models.yolov2.anchor import PriorBox
from ..models.yolov2.loss import YOLOV2Loss
from ..models.yolov2.model import YOLOV2

BATCH_SIZE = 2
IMAGE_SIZE = 416
GRID_SIZE = 13
MAX_OBJECT_SIZE = 10
NUM_CLASS = 20


CONTEXT = DetectorContext(
    detector_name="yolov2",
    dataset="voc",
    device="cpu",
    num_classes=NUM_CLASS,
    num_anchors=5,
    anchors_path=__file__.replace("test_yolov2.py", "fake.anchors.yolov2.txt"),
)

OUTPUT_SIZE = (CONTEXT.num_classes + 5) * CONTEXT.num_anchors


class TestModel:
    def test_yolov2_output_shape(self):
        backbone = DarkNet19FeatureExtractor("last_conv", trainable=False)
        backbone.fix_target_layers(["3_1", "4_1"])

        model = YOLOV2(
            backbone,
            CONTEXT,
            passthrough_feature_dim=backbone.feature_dims[-2],
            neck_feature_dim=backbone.feature_dims[-1],
        )
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        output: torch.Tensor = model(x)
        assert output.shape == torch.Size(
            (
                BATCH_SIZE,
                OUTPUT_SIZE,
                GRID_SIZE,
                GRID_SIZE,
            )
        )

    @pytest.mark.parametrize("network_name", ["resnet18", "resnet34", "resnet50"])
    def test_yolov2_resnet_output_shape(self, network_name: str):
        backbone = ResNetFeatureExtractor(network_name, "block", False)
        backbone.fix_target_layers(["3_1", "4_1"])

        model = YOLOV2(
            backbone,
            CONTEXT,
            passthrough_feature_dim=backbone.feature_dims[-2],
            neck_feature_dim=backbone.feature_dims[-1],
        )
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        output: torch.Tensor = model(x)
        assert output.shape == torch.Size(
            (
                BATCH_SIZE,
                OUTPUT_SIZE,
                GRID_SIZE,
                GRID_SIZE,
            )
        )


class TestLoss:
    def setUp(self):
        backbone = ResNetFeatureExtractor("resnet18", "block", False)
        backbone.fix_target_layers(["3_1", "4_1"])

        model = YOLOV2(
            backbone,
            CONTEXT,
            passthrough_feature_dim=backbone.feature_dims[-2],
            neck_feature_dim=backbone.feature_dims[-1],
        )
        model.requires_grad_()
        self.yhat = model(
            torch.rand(
                BATCH_SIZE,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )
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
        priorBox = PriorBox(CONTEXT)
        self.anchors = priorBox.load_anchors()

    def test_match(self):
        self.setUp()

        criterion = YOLOV2Loss(self.anchors, CONTEXT, iou_threshold=0)
        y, span = build_flatten_targets(self.y, (GRID_SIZE, GRID_SIZE))
        pred = torch.rand(
            BATCH_SIZE,
            CONTEXT.num_anchors,
            5 + CONTEXT.num_classes,
            GRID_SIZE,
            GRID_SIZE,
        )

        with torch.no_grad():
            target, positivity = criterion.match(y, span, pred, GRID_SIZE, GRID_SIZE)
        # assert (iou - labels[:, :, 4:5, :, :]).sum() < 1e-2, "iou is too far away"
        assert len(target) == BATCH_SIZE
        assert torch.cat(target).shape == torch.Size((sum(span), 7))
        assert positivity.shape == torch.Size(
            (BATCH_SIZE, CONTEXT.num_anchors * GRID_SIZE * GRID_SIZE)
        )

    @pytest.mark.parametrize("seen", [100, 20000])
    def test_forward(self, seen: int):
        self.setUp()

        criterion = YOLOV2Loss(self.anchors, CONTEXT)

        loss = criterion(self.yhat, self.y, seen=seen)
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("seen", [100, 20000])
    def test_backward(self, seen: int):
        self.setUp()

        criterion = YOLOV2Loss(self.anchors, CONTEXT)

        loss = criterion(self.yhat, self.y, seen=seen)
        loss.backward()
