import random

import pytest
import torch

from torchlake.common.models import ResNetFeatureExtractor
from torchlake.image_classification.models.darknet53 import DarkNet53FeatureExtractor

from ..constants.schema import DetectorContext
from ..models.yolov3.anchor import PriorBox
from ..models.yolov3.loss import YOLOV3Loss
from ..models.yolov3.model import YOLOV3
from ..models.yolov3.network import SPP, RegHead
from ..utils.train import build_flatten_targets

BATCH_SIZE = 2
IMAGE_SIZE = 416
MAX_OBJECT_SIZE = 10
NUM_CLASS = 20
HIDDEN_DIM = 8

CONTEXT = DetectorContext(
    detector_name="yolov3",
    dataset="voc",
    device="cpu",
    num_classes=NUM_CLASS,
    num_anchors=(3, 3, 3),
    grid_sizes=(IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8),
    anchors_path=__file__.replace("test_yolov3.py", "fake.anchors.yolov3.txt"),
)

OUTPUT_SIZE = CONTEXT.num_classes + 5


NUM_BOXES = tuple(
    num_anchor * grid_size**2
    for num_anchor, grid_size in zip(CONTEXT.num_anchors, CONTEXT.grid_sizes)
)


class TestModel:
    @pytest.mark.parametrize("enable_spp", (True, False))
    def test_yolov3_forward_shape(self, enable_spp: bool):
        backbone = DarkNet53FeatureExtractor("block", trainable=False)
        backbone.fix_target_layers(["2_1", "3_1", "4_1"])

        model = YOLOV3(
            backbone,
            CONTEXT,
            hidden_dim_8x=backbone.feature_dims[-3],
            hidden_dim_16x=backbone.feature_dims[-2],
            hidden_dim_32x=backbone.feature_dims[-1],
            enable_spp=enable_spp,
        )
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        outputs: tuple[torch.Tensor] = model(x)
        for output, num_anchor, grid_size in zip(
            outputs,
            CONTEXT.num_anchors,
            CONTEXT.grid_sizes,
        ):
            assert output.shape == torch.Size(
                (
                    BATCH_SIZE,
                    num_anchor * OUTPUT_SIZE,
                    grid_size,
                    grid_size,
                )
            )

    @pytest.mark.parametrize("network_name", ["resnet18", "resnet34", "resnet50"])
    @pytest.mark.parametrize("enable_spp", (True, False))
    def test_yolov3_resnet_forward_shape(self, network_name: str, enable_spp: bool):
        backbone = ResNetFeatureExtractor(network_name, "block", trainable=False)
        backbone.fix_target_layers(["2_1", "3_1", "4_1"])

        model = YOLOV3(
            backbone,
            CONTEXT,
            hidden_dim_8x=backbone.feature_dims[-3],
            hidden_dim_16x=backbone.feature_dims[-2],
            hidden_dim_32x=backbone.feature_dims[-1],
            enable_spp=enable_spp,
        )
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        outputs: tuple[torch.Tensor] = model(x)
        for output, num_anchor, grid_size in zip(
            outputs,
            CONTEXT.num_anchors,
            CONTEXT.grid_sizes,
        ):
            assert output.shape == torch.Size(
                (
                    BATCH_SIZE,
                    num_anchor * OUTPUT_SIZE,
                    grid_size,
                    grid_size,
                )
            )


class TestNetwork:
    def test_reghead_forward_shape(self):
        grid_size = CONTEXT.grid_sizes[0]
        num_anchor = CONTEXT.num_anchors[0]

        model = RegHead(HIDDEN_DIM, num_anchor, CONTEXT.num_classes)

        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, grid_size, grid_size)
        y = model(x)

        assert y.shape == torch.Size(
            [
                BATCH_SIZE,
                num_anchor * OUTPUT_SIZE,
                grid_size,
                grid_size,
            ]
        )

    def test_spp_forward_shape(self):
        grid_size = CONTEXT.grid_sizes[0]

        model = SPP()

        x = torch.rand(BATCH_SIZE, HIDDEN_DIM, grid_size, grid_size)
        y = model(x)

        assert y.shape == torch.Size(
            [
                BATCH_SIZE,
                HIDDEN_DIM * (1 + len(model.kernel_sizes)),
                grid_size,
                grid_size,
            ]
        )


class TestLoss:
    def setUp(self):
        self.grid_size = CONTEXT.grid_sizes[0]

        backbone = ResNetFeatureExtractor("resnet18", "block", False)
        backbone.fix_target_layers(["2_1", "3_1", "4_1"])

        model = YOLOV3(
            backbone,
            CONTEXT,
            hidden_dim_8x=backbone.feature_dims[-3],
            hidden_dim_16x=backbone.feature_dims[-2],
            hidden_dim_32x=backbone.feature_dims[-1],
            enable_spp=False,
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
                    random.random(),
                    random.random(),
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
        num_anchors = CONTEXT.num_anchors[0]

        criterion = YOLOV3Loss(self.anchors, CONTEXT, iou_threshold=0)
        y, span = build_flatten_targets(
            self.y,
            (self.grid_size, self.grid_size),
            delta_coord=True,
        )
        pred = torch.rand(
            BATCH_SIZE,
            num_anchors,
            OUTPUT_SIZE,
            self.grid_size,
            self.grid_size,
        )

        with torch.no_grad():
            best_prior_indices = criterion.match_anchor(self.y)
            target, positivity = criterion.match(
                y,
                span,
                pred,
                best_prior_indices,
                self.grid_size,
                self.grid_size,
                tuple(range(num_anchors)),
            )
        # assert (iou - labels[:, :, 4:5, :, :]).sum() < 1e-2, "iou is too far away"
        assert len(target) == BATCH_SIZE
        assert torch.cat(target).shape == torch.Size((sum(span), 7))
        assert positivity.shape == torch.Size((BATCH_SIZE, NUM_BOXES[0]))

    @pytest.mark.parametrize("cls_loss_type", ["sigmoid", "softmax"])
    @pytest.mark.parametrize("loc_loss_type", ["mse", "diou", "ciou", "giou"])
    def test_forward(self, cls_loss_type: str, loc_loss_type: str):
        self.setUp()

        criterion = YOLOV3Loss(
            self.anchors,
            CONTEXT,
            cls_loss_type=cls_loss_type,
            loc_loss_type=loc_loss_type,
        )

        loss = criterion(self.yhat, self.y)
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("cls_loss_type", ["sigmoid", "softmax"])
    @pytest.mark.parametrize("loc_loss_type", ["mse", "diou", "ciou", "giou"])
    def test_backward(self, cls_loss_type: str, loc_loss_type: str):
        self.setUp()

        criterion = YOLOV3Loss(
            self.anchors,
            CONTEXT,
            cls_loss_type=cls_loss_type,
            loc_loss_type=loc_loss_type,
        )

        loss = criterion(self.yhat, self.y)
        loss.backward()
