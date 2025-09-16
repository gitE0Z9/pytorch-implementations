import torch

from ..constants.schema import DetectorContext
from ..models.yolov3_tiny.model import YOLOV3Tiny

BATCH_SIZE = 2
IMAGE_SIZE = 416
MAX_OBJECT_SIZE = 100
CONTEXT = DetectorContext(
    detector_name="yolov3-tiny",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=(3, 3),
    grid_sizes=(IMAGE_SIZE // 16, IMAGE_SIZE // 64),
    anchors_path=__file__.replace(
        "test_yolov3_tiny.py",
        "fake.anchors.yolov3_tiny.txt",
    ),
)


class TestModel:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        model = YOLOV3Tiny(CONTEXT)

        feature_maps = model(x)

        for feature_map, num_anchor, grid_size in zip(
            feature_maps, CONTEXT.num_anchors, CONTEXT.grid_sizes
        ):
            assert feature_map.shape == torch.Size(
                (
                    BATCH_SIZE,
                    num_anchor * (4 + 1 + CONTEXT.num_classes),
                    grid_size,
                    grid_size,
                ),
            )
