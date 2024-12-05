import random
import torch
from torch.testing import assert_close
from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.extraction import ExtractionFeatureExtractor

from ..constants.schema import DetectorContext
from ..models.yolov1.loss import YOLOLoss
from ..models.yolov1.model import YOLOV1, YOLOV1Modified

BATCH_SIZE = 2
GRID_SIZE = 7
MAX_OBJECT_SIZE = 10
NUM_CLASS = 20

CONTEXT = DetectorContext(
    detector_name="yolov1",
    dataset="voc",
    device="cpu",
    num_classes=NUM_CLASS,
    num_anchors=2,
    anchors_path="",
)

OUTPUT_SIZE = CONTEXT.num_classes + CONTEXT.num_anchors * 5


class TestYOLOV1Original:
    def test_output_shape(self):
        backbone = ExtractionFeatureExtractor("block", trainable=False)
        backbone.fix_target_layers(["3_1"])

        model = YOLOV1(backbone, CONTEXT)
        x = torch.rand(2, 3, 448, 448)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, OUTPUT_SIZE, 7, 7]))


class TestYOLOV1Modified:
    def test_output_shape(self):
        backbone = ResNetFeatureExtractor("resnet18", "block", False)
        backbone.fix_target_layers(["4_1"])

        model = YOLOV1Modified(backbone, CONTEXT)
        x = torch.rand(2, 3, 448, 448)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, OUTPUT_SIZE, 7, 7]))


class TestYOLOLoss:
    def setUp(self):
        self.x = torch.rand(
            BATCH_SIZE,
            OUTPUT_SIZE,
            GRID_SIZE,
            GRID_SIZE,
        ).requires_grad_()
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

    def test_isnan(self):
        self.setUp()

        criterion = YOLOLoss(CONTEXT)

        loss = criterion(self.x, self.y)
        assert not torch.isnan(loss)

    def test_backward(self):
        self.setUp()

        criterion = YOLOLoss(CONTEXT)

        loss = criterion(self.x, self.y)
        loss.backward()
