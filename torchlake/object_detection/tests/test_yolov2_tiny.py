import torch

from ..constants.schema import DetectorContext
from ..models.yolov2_tiny.model import YOLOV2Tiny

BATCH_SIZE = 2
IMAGE_SIZE = 416
GRID_SIZE = 6
MAX_OBJECT_SIZE = 100
CONTEXT = DetectorContext(
    detector_name="yolov2-tiny",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=5,
    anchors_path=__file__.replace("test_yolov2.py", "fake.anchors.txt"),
)


class TestModel:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        model = YOLOV2Tiny(CONTEXT)

        y = model(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                CONTEXT.num_anchors * (4 + 1 + CONTEXT.num_classes),
                GRID_SIZE,
                GRID_SIZE,
            )
        )
