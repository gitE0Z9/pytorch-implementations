import torch
from torch.testing import assert_close
from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.extraction import ExtractionFeatureExtractor

from ..constants.schema import DetectorContext
from ..models.yolov1.model import YOLOV1, YOLOV1Modified

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
