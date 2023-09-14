import torch
from torch.testing import assert_equal

from utils.train import (
    generate_grid_train,
    xywh_to_xyxy,
    IOU,
    collate_fn,
    build_targets,
)


class TestUtilsTrain:
    def test_generate_grid_train_shape(self):
        """not useful test"""
        grid = generate_grid_train(13, 13)
        assert_equal(grid.shape, torch.Size([1, 1, 2, 13, 13]))

    def test_xywh_to_xyxy_shape(self):
        """not useful test"""
        testx = torch.rand((2, 5, 4, 13, 13))
        convert_x = xywh_to_xyxy(testx)
        assert_equal(testx.shape, convert_x.shape)

    def test_iou_shape(self):
        """not useful test"""
        testx = torch.rand((2, 5, 4, 13, 13))
        testy = torch.rand((2, 1, 4, 13, 13))
        ious = IOU(testx, testy)
        assert_equal(ious.shape, torch.Size([2, 5, 1, 13, 13]))
