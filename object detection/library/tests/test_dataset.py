import random

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.testing import assert_close
from datasets.voc.datasets import VOCDatasetRaw, VOCDatasetFromCSV
from datasets.coco.datasets import COCODatasetRaw, COCODatasetFromCSV
from datasets.imagenet.datasets import ImageNetDataset

# class TestVOCDatasetRaw():

#     def test_get_img(self):
#         trainset = VOC_dataset_raw(root=cfg['DATA']['VOC']['ROOT'], class_name=cls_name,
#                                 year='2012+2007', mode='trainval')

#         pick = random.randint(0, len(trainset))
#         img, h, w = trainset.get_img(pick)

#         assert img.shape == (h,w,3)

#     def test_get_label(self):
#         trainset = VOC_dataset_raw(root=cfg['DATA']['VOC']['ROOT'], class_name=cls_name,
#                                 year='2012+2007', mode='trainval')

#         pick = random.randint(0, len(trainset))
#         _, h, w = trainset.get_img(pick)
#         label = trainset.get_label(pick,h,w)

#         assert all(isinstance(l[4], int) for l in label)
#         for i in range(4):
#             assert all(isinstance(l[i], float) for l in label)

#     def test_getitem(self):
#         trainset = VOC_dataset_raw(root=cfg['DATA']['VOC']['ROOT'], class_name=cls_name,
#                                 year='2012+2007', mode='trainval',transform=preprocess)

#         pick = random.randint(0, len(trainset))
#         img, label = trainset[pick]

#         assert_equal(img.shape, torch.Size([3, 416, 416]))
#         assert all(isinstance(l[4], int) for l in label)
#         for i in range(4):
#             assert all(isinstance(l[0], float) for l in label)


class TestVOCDatasetFromCSV:    
    def test_get_img(self):
        dataset = VOCDatasetFromCSV()

        pick = random.randint(0, len(dataset))
        _, path = dataset.get_label(pick)
        img = dataset.get_img(path)

        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert img.shape[2] == 3

    def test_get_label(self):
        dataset = VOCDatasetFromCSV()

        pick = random.randint(0, len(dataset))
        label, _ = dataset.get_label(pick)

        for l in label:
            assert all(isinstance(l[i], float) for i in range(4))

    def test_get_item(self):
        preprocess = A.Compose(
            [
                A.Resize(416, 416),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo"),
        )
        dataset = VOCDatasetFromCSV(transform=preprocess)

        pick = random.randint(0, len(dataset))
        img, label = dataset[pick]

        assert_close(img.shape, torch.Size([3, 416, 416]))
        for l in label:
            assert all(isinstance(l[i], float) for i in range(4))
