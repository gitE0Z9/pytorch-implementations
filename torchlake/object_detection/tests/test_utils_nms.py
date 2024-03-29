import torch
from torch.testing import assert_close

from ..configs.schema import InferenceCfg
from ..utils.nms import select_best_index


class TestNms:
    def test_overlap(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1], [0.25, 0.25, 0.5, 0.5, 0.9]])
        cfg = InferenceCfg(METHOD="torchvision", CONF_THRESH=0.3, NMS_THRESH=0.5)

        indices = select_best_index(x[:, :4], x[:, 4], cfg)

        assert len(indices) == 1
        assert_close(torch.from_numpy(indices), torch.LongTensor([0]))

    def test_non_overlap(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1], [0.75, 0.75, 0.5, 0.5, 0.9]])
        cfg = InferenceCfg(METHOD="torchvision", CONF_THRESH=0.3, NMS_THRESH=0.5)

        indices = select_best_index(x[:, :4], x[:, 4], cfg)

        assert len(indices) == 2
        assert_close(torch.from_numpy(indices), torch.LongTensor([0, 1]))

    def test_conf_lower(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1], [0.75, 0.75, 0.5, 0.5, 0.1]])
        cfg = InferenceCfg(METHOD="torchvision", CONF_THRESH=0.3, NMS_THRESH=0.5)

        indices = select_best_index(x[:, :4], x[:, 4], cfg)

        assert len(indices) == 1
        assert_close(torch.from_numpy(indices), torch.LongTensor([0]))

    def test_iou_half(self):
        x = torch.Tensor([[0.25, 0.25, 0.5, 0.5, 1], [0.4, 0.25, 0.5, 0.5, 0.9]])
        cfg = InferenceCfg(METHOD="torchvision", CONF_THRESH=0.3, NMS_THRESH=0.5)

        indices = select_best_index(x[:, :4], x[:, 4], cfg)

        assert len(indices) == 1
        assert_close(torch.from_numpy(indices), torch.LongTensor([0]))
