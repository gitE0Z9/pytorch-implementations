import torch
from torch.testing import assert_close

from ..constants.schema import DetectorContext
from ..models.yolov1_tiny.model import YOLOV1Tiny

BATCH_SIZE = 2
IMAGE_SIZE = 448
GRID_SIZE = 7
MAX_OBJECT_SIZE = 100
CONTEXT = DetectorContext(
    detector_name="yolov1-tiny",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=2,
    anchors_path="",
)


class TestYOLOV1Tiny:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        model = YOLOV1Tiny(CONTEXT)

        y = model(x)

        assert_close(
            y.shape,
            torch.Size(
                (
                    BATCH_SIZE,
                    CONTEXT.num_anchors * (4 + 1) + CONTEXT.num_classes,
                    GRID_SIZE,
                    GRID_SIZE,
                )
            ),
        )
