import torch
from torch.testing import assert_close
from torchlake.common.models import ResNetFeatureExtractor

from ..constants.schema import DetectorContext
from ..models.yolov1.model import YOLOV1Modified, YOLOV1Original

CONTEXT = DetectorContext(
    detector_name="yolov1",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=2,
    anchors_path="",
)

OUTPUT_SIZE = CONTEXT.num_classes + CONTEXT.num_anchors * 5


class TestYOLOV1Original:
    def test_output_shape(self):
        backbone = ResNetFeatureExtractor("resnet18", "block", False)
        model = YOLOV1Original(backbone, CONTEXT)
        x = torch.rand(2, 3, 448, 448)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, OUTPUT_SIZE, 7, 7]))


class TestYOLOV1Modified:
    def test_output_shape(self):
        backbone = ResNetFeatureExtractor("resnet18", "block", False)
        model = YOLOV1Modified(backbone, CONTEXT)
        x = torch.rand(2, 3, 448, 448)

        output: torch.Tensor = model(x)
        assert_close(output.shape, torch.Size([2, OUTPUT_SIZE, 7, 7]))