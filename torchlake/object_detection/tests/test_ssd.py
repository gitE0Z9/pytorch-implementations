import random
import torch
from torch.testing import assert_close

from ..constants.schema import DetectorContext
from ..models.ssd.anchor import PriorBox
from ..models.ssd.decode import Decoder
from ..models.ssd.loss import MultiBoxLoss
from ..models.ssd.model import SSD
from ..models.ssd.network import Backbone, RegHead

BATCH_SIZE = 2
IMAGE_SIZE = 300  # 512
GRID_SIZE = 7  # XXX
MAX_OBJECT_SIZE = 100
CONTEXT = DetectorContext(
    detector_name="ssd",
    dataset="voc",
    device="cpu",
    num_classes=20,
    num_anchors=[4, 6, 6, 6, 4, 4],
    anchors_path="",
)

TOTAL_ANCHORS = 8732


class TestBackbone:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        model = Backbone()

        features = model(x)
        expected_shapes = (
            torch.Size((BATCH_SIZE, dim, scale, scale))
            for dim, scale in zip(
                (512, 1024, 512, 256, 256, 256),
                (38, 19, 10, 5, 3, 1),
            )
        )

        for feature, expected_shape in zip(features, expected_shapes):
            assert_close(feature.shape, expected_shape)


class TestRegHead:
    def test_output_shape(self):
        x = torch.rand(BATCH_SIZE, 16, 7, 7)

        model = RegHead(16, CONTEXT.num_classes + 1, 5, 4)

        y = model(x)
        assert_close(
            y.shape,
            torch.Size((BATCH_SIZE, 5 * (4 + 1 + CONTEXT.num_classes), 7, 7)),
        )


class TestSSD:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        model = SSD(CONTEXT)

        y = model(x)

        assert_close(
            y.shape,
            torch.Size((BATCH_SIZE, 8732, 4 + 1 + CONTEXT.num_classes)),
        )


class TestPriorBox:
    def test_build_anchors_shape(self): ...


class TestDecoder:
    def test_decode_shape(self): ...
    def test_post_process_shape(self): ...


class TestMultiBoxLoss:
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
        criterion = MultiBoxLoss(CONTEXT, anchors)
        loss = criterion(self.pred, self.y)

        assert not torch.isnan(loss)

    def test_backward(self):
        self.setUp()

        anchors = PriorBox().build_anchors()
        criterion = MultiBoxLoss(CONTEXT, anchors)
        loss = criterion(self.pred, self.y)

        loss.backward()