import torch
from configs.schema import Setting
from utils.config import load_anchors, load_classes, load_config


class Testconfig:
    def test_anchors(self):
        anchors = load_anchors("configs/yolov2/anchors.txt")
        cfg = load_config("configs/yolov2/resnet18.yml")
        assert anchors.shape == torch.Size([1, cfg["MODEL"]["NUM_ANCHORS"], 2, 1, 1])

    def test_classes(self):
        cls_name = load_classes("datasets/voc/voc_classes.txt")
        assert len(cls_name) == 20, "wrong length"

    def test_resnet18_config(self):
        cfg = load_config("configs/yolov2/resnet18.yml")
        Setting(**cfg)

    def test_resnet34_config(self):
        cfg = load_config("configs/yolov2/resnet34.yml")
        Setting(**cfg)

    def test_resnet50_config(self):
        cfg = load_config("configs/yolov2/resnet50.yml")
        Setting(**cfg)

    def test_darknet19_config(self):
        cfg = load_config("configs/yolov2/darknet19.yml")
        Setting(**cfg)
