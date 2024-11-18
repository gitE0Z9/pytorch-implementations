import random
import pytest
import torch
from torch.testing import assert_close

from ..utils.train import (
    IOU,
    collate_fn,
    generate_grid_train,
    xywh_to_xyxy,
    build_flatten_targets,
    build_grid_targets,
)

BATCH_SIZE = 2
GRID_SIZE = 7
MAX_OBJECT_SIZE = 10
NUM_CLASS = 20


class TestUtilsTrain:
    def test_generate_grid_train_shape(self):
        grid = generate_grid_train(13, 13)

        assert_close(grid.shape, torch.Size([1, 1, 2, 13, 13]))

    def test_xywh_to_xyxy_shape(self):
        testx = torch.rand((2, 5, 4, 13, 13))

        convert_x = xywh_to_xyxy(testx)
        assert_close(testx.shape, convert_x.shape)

    def test_xywh_to_xyxy_overlap(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1], [0.25, 0.25, 0.5, 0.5, 0.9]]).view(
            2, 1, 5, 1, 1
        )

        convert_x = xywh_to_xyxy(x[:, :4])
        should_x = torch.Tensor([[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]]).view(
            2, 1, 4, 1, 1
        )
        assert_close(convert_x, should_x)

    def test_xywh_to_xyxy_non_overlap(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1], [0.75, 0.75, 0.5, 0.5, 0.9]]).view(
            2, 1, 5, 1, 1
        )

        convert_x = xywh_to_xyxy(x[:, :4])
        should_x = torch.Tensor([[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]).view(
            2, 1, 4, 1, 1
        )
        assert_close(convert_x, should_x)

    def test_iou_shape(self):
        testx = torch.rand((2, 5, 4, 13, 13))
        testy = torch.rand((2, 1, 4, 13, 13))

        ious = IOU(testx, testy)
        assert_close(ious.shape, torch.Size([2, 5, 1, 13, 13]))

    def test_iou_overlap(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1]]).view(1, 1, 5, 1, 1)
        y = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 0.9]]).view(1, 1, 5, 1, 1)

        ious = IOU(x, y)
        assert_close(ious, torch.ones(1, 1, 1, 1, 1))

    def test_iou_non_overlap(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1]]).view(1, 1, 5, 1, 1)
        y = torch.Tensor([[0.75, 0.75, 0.5, 0.5, 0.9]]).view(1, 1, 5, 1, 1)

        ious = IOU(x, y)
        assert_close(ious, torch.zeros(1, 1, 1, 1, 1))


class TestBuildTargets:
    def setUp(self):
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

    @pytest.mark.parametrize(
        "grid_shape,delta_coord,expected_dim",
        [[(GRID_SIZE, GRID_SIZE), True, 7], [None, False, 5]],
    )
    def test_build_flatten_targets(self, grid_shape, delta_coord, expected_dim):
        self.setUp()

        y, span = build_flatten_targets(self.y, grid_shape, delta_coord)

        N = sum(len(items) for items in self.y)
        assert_close(y.shape, torch.Size((N, expected_dim)))
        assert len(span) == BATCH_SIZE

    def test_build_grid_targets(self):
        self.setUp()

        expected_shape = (BATCH_SIZE, 1, NUM_CLASS + 5, GRID_SIZE, GRID_SIZE)
        y = build_grid_targets(self.y, expected_shape)

        assert_close(y.shape, torch.Size(expected_shape))
