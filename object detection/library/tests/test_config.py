from configs.schema import Setting
from utils.config import load_classes, load_config
from datasets.schema import DatasetCfg


class TestSSDConfig:
    def test_vgg16_config(self):
        cfg = load_config("configs/ssd/vgg16.yml")
        Setting(**cfg)


class TestYOLOv1Config:
    def test_extraction_config(self):
        cfg = load_config("configs/yolov1/extraction.yml")
        Setting(**cfg)

    def test_resnet18_config(self):
        cfg = load_config("configs/yolov1/resnet18.yml")
        Setting(**cfg)

    def test_resnet34_config(self):
        cfg = load_config("configs/yolov1/resnet34.yml")
        Setting(**cfg)

    def test_resnet50_config(self):
        cfg = load_config("configs/yolov1/resnet50.yml")
        Setting(**cfg)


class TestYOLOv2Config:
    # def test_anchors(self):
    #     anchors = load_anchors("configs/yolov2/anchors.txt")
    #     cfg = load_config("configs/yolov2/resnet18.yml")
    #     assert anchors.shape == torch.Size([1, cfg["MODEL"]["NUM_ANCHORS"], 2, 1, 1])

    def test_darknet19_config(self):
        cfg = load_config("configs/yolov2/darknet19.yml")
        Setting(**cfg)

    def test_resnet18_config(self):
        cfg = load_config("configs/yolov2/resnet18.yml")
        Setting(**cfg)

    def test_resnet34_config(self):
        cfg = load_config("configs/yolov2/resnet34.yml")
        Setting(**cfg)

    def test_resnet50_config(self):
        cfg = load_config("configs/yolov2/resnet50.yml")
        Setting(**cfg)


class TestDatasetConfig:
    voc_cfg_path = "datasets/voc/config.yml"
    voc_classes_path = "datasets/voc/voc_classes.txt"

    coco_cfg_path = "datasets/coco/config.yml"
    coco_classes_path = "datasets/coco/coco_classes.txt"

    imagenet_cfg_path = "datasets/imagenet/config.yml"
    imagenet_classes_path = "datasets/imagenet/imagenet_classes.txt"

    def test_config_voc(self):
        cfg = load_config(self.voc_cfg_path)
        DatasetCfg(**cfg)

    def test_config_coco(self):
        cfg = load_config(self.coco_cfg_path)
        DatasetCfg(**cfg)

    def test_config_imagenet(self):
        cfg = load_config(self.imagenet_cfg_path)
        DatasetCfg(**cfg)

    def test_num_classes_voc(self):
        class_names = load_classes(self.voc_classes_path)
        num_classes = load_config(self.voc_cfg_path)["NUM_CLASSES"]
        assert len(class_names) == num_classes, "wrong length"

    def test_num_classes_coco(self):
        cls_name = load_classes(self.coco_classes_path)
        num_classes = load_config(self.coco_cfg_path)["NUM_CLASSES"]
        assert len(cls_name) == num_classes, "wrong length"

    def test_num_classes_imagenet(self):
        cls_name = load_classes(self.imagenet_classes_path)
        num_classes = load_config(self.imagenet_cfg_path)["NUM_CLASSES"]
        assert len(cls_name) == num_classes, "wrong length"
