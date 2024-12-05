import random
import torch
from torch.testing import assert_close

from ..constants.schema import DetectorContext
from ..models.retinanet.anchor import PriorBox
from torchlake.common.models import ResNetFeatureExtractor

# from ..models.retinanet.decode import Decoder
from ..models.retinanet.loss import FocalLoss, RetinaNetLoss
from ..models.retinanet.model import RetinaNet
from ..models.retinanet.network import RegHead

BATCH_SIZE = 2
IMAGE_SIZE = 600
MAX_OBJECT_SIZE = 100
CONTEXT = DetectorContext(
    detector_name="retinanet",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=9,
    anchors_path="",
)

TOTAL_ANCHORS = 68670


class TestRegHead:
    def test_output_shape(self):
        x = torch.rand(BATCH_SIZE, 16, 7, 7)

        model = RegHead(
            16,
            num_priors=9,
            num_classes=CONTEXT.num_classes + 1,
        )

        y = model(x)
        assert_close(
            y.shape,
            torch.Size(
                (BATCH_SIZE, CONTEXT.num_anchors * (4 + 1 + CONTEXT.num_classes), 7, 7)
            ),
        )


class TestRetinaNet:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet18", "block", False)
        backbone.fix_target_layers(["2_1", "3_1", "4_1"])
        model = RetinaNet(backbone, CONTEXT)

        y = model(x)

        assert_close(
            y.shape,
            torch.Size((BATCH_SIZE, 68670, 4 + 1 + CONTEXT.num_classes)),
        )


class TestFocalLoss:
    def setUp(self):
        self.y = torch.randint(0, CONTEXT.num_classes, (BATCH_SIZE,))
        self.pred = torch.rand((BATCH_SIZE, CONTEXT.num_classes)).requires_grad_()

    def test_forward(self):
        self.setUp()

        criterion = FocalLoss()
        loss = criterion(self.pred, self.y)

        assert not torch.isnan(loss)

    def test_backward(self):
        self.setUp()

        criterion = FocalLoss()
        loss = criterion(self.pred, self.y)

        loss.backward()


class TestRetinaNetLoss:
    def setUp(self):
        self.y = [
            [
                [
                    random.random(),
                    random.random(),
                    random.random() * 0.1,
                    random.random() * 0.1,
                    random.randrange(0, CONTEXT.num_classes),
                ]
                for _ in range(random.randint(1, MAX_OBJECT_SIZE))
            ]
            for _ in range(BATCH_SIZE)
        ]
        self.pred = torch.rand(
            (BATCH_SIZE, TOTAL_ANCHORS, 4 + 1 + CONTEXT.num_classes)
        ).requires_grad_()

    def test_forward(self):
        self.setUp()

        anchors = PriorBox().build_anchors()
        criterion = RetinaNetLoss(CONTEXT, anchors)
        loss = criterion(self.pred, self.y)

        assert not torch.isnan(loss)

    def test_backward(self):
        self.setUp()

        anchors = PriorBox().build_anchors()
        criterion = RetinaNetLoss(CONTEXT, anchors)
        loss = criterion(self.pred, self.y)

        loss.backward()
