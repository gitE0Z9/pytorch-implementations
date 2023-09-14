import random

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.testing import assert_equal
from tqdm import tqdm
from datasets.voc.datasets import VOCDatasetFromCSV
from models.yolov2.detector import Yolov2Resnet
from models.yolov2.loss import YOLOv2Loss
from utils.config import load_classes, load_config
from utils.train import collate_fn, build_targets

cfg = load_config("config/resnet18.yml")
cls_name = load_classes("config/voc_classes.txt")

preprocess = A.Compose(
    [
        A.Resize(416, 416),
        A.ColorJitter(),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # changeable
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_area=1024, min_visibility=0.3),
)

model = Yolov2Resnet(18, 5, 20)
criterion = YOLOv2Loss(5, device="cpu")
dataset = VOCDatasetFromCSV(mode="trainval", transform=preprocess)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)


class TestYOLOv2Loss:
    def test_iou_box(self):
        count = 0
        for _, labels in loader:
            labels = build_targets(labels, (2, 1, 25, 13, 13))
            iou, best_box = criterion.iou_box(labels, labels)
            assert (iou - labels[:, :, 4:5, :, :]).sum() < 1e-2, "iou is too far away"
            assert_equal(best_box.shape, torch.Size([2, 1, 1, 13, 13]))
            count += 1
            if count > 5:
                break

    def test_build_targets(self):
        """not a good test"""
        count = 0
        for _, labels in loader:
            labels = build_targets(labels, (2, 1, 25, 13, 13))
            assert_equal(labels.shape, torch.Size([2, 1, 25, 13, 13]))
            count += 1
            if count > 5:
                break

    def test_forward(self):
        max_batch = 20
        count = 0
        seen = 0
        for img, label in loader:
            seen += img.size(0)
            output = model(img)
            loss = criterion(output, label, anchors, seen)
            assert not torch.isnan(loss), "forward failed, loss died"
            count += 1
            if count > max_batch:
                break

    def test_backward(self):
        max_batch = 20
        count = 0
        seen = 0
        for img, label in loader:
            seen += img.size(0)
            output = model(img)
            loss = criterion(output, label, anchors, seen)
            loss.backward()
            assert not torch.isnan(loss), "forward failed, loss died"
            count += 1
            if count > max_batch:
                break
