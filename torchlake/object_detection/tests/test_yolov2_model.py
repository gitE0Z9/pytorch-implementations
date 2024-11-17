import pytest
import torch
from torch.testing import assert_close
from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.darknet19 import DarkNet19FeatureExtractor

from ..constants.schema import DetectorContext
from ..models.yolov2.model import YOLOV2

CONTEXT = DetectorContext(
    detector_name="yolov2",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=5,
    anchors_path="",
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
