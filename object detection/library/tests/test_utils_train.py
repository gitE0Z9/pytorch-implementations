import torch
from torch.testing import assert_close

from utils.train import (
    generate_grid_train,
    xywh_to_xyxy,
    IOU,
    collate_fn,
    build_targets,
)


class TestUtilsTrain:
    def test_generate_grid_train_shape(self):
        grid = generate_grid_train(13, 13)
        
        assert_close(grid.shape, torch.Size([1, 1, 2, 13, 13]))

    def test_xywh_to_xyxy_shape(self):
        testx = torch.rand((2, 5, 4, 13, 13))
        
        convert_x = xywh_to_xyxy(testx)
        assert_close(testx.shape, convert_x.shape)
        
    def test_xywh_to_xyxy_overlap(self):
        x = torch.Tensor([[.25, .25, .5, .5, 1], [.25, .25, .5, .5, .9]]).view(2, 1, 5, 1, 1)
        
        convert_x = xywh_to_xyxy(x[:, :4])
        should_x = torch.Tensor([[0, 0, .5, .5], [0, 0, .5, .5]]).view(2, 1, 4, 1, 1)
        assert_close(convert_x, should_x)
        
    def test_xywh_to_xyxy_non_overlap(self):
        x = torch.Tensor([[.25, .25, .5, .5, 1], [.75, .75, .5, .5, .9]]).view(2, 1, 5, 1, 1)
        
        convert_x = xywh_to_xyxy(x[:, :4])
        should_x = torch.Tensor([[0, 0, .5, .5], [.5, .5, 1, 1]]).view(2, 1, 4, 1, 1)
        assert_close(convert_x, should_x)

    def test_iou_shape(self):
        testx = torch.rand((2, 5, 4, 13, 13))
        testy = torch.rand((2, 1, 4, 13, 13))
        
        ious = IOU(testx, testy)
        assert_close(ious.shape, torch.Size([2, 5, 1, 13, 13]))
        
    def test_iou_overlap(self):
        x = torch.Tensor([[.25, .25, .5, .5, 1]]).view(1, 1, 5, 1, 1)
        y = torch.Tensor([[.25, .25, .5, .5, .9]]).view(1, 1, 5, 1, 1)
        
        ious = IOU(x, y)
        assert_close(ious, torch.Tensor([1]).view(1, 1, 1, 1, 1))
        
    def test_iou_non_overlap(self):
        x = torch.Tensor([[.25, .25, .5, .5, 1]]).view(1, 1, 5, 1, 1)
        y = torch.Tensor([[.75, .75, .5, .5, .9]]).view(1, 1, 5, 1, 1)
        
        ious = IOU(x, y)
        assert_close(ious, torch.Tensor([0]).view(1, 1, 1, 1, 1))
