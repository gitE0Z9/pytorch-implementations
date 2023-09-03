import random

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.testing import assert_equal
from tqdm import tqdm
from yolov2.datasets import VOC_dataset_raw, VOC_dataset_csv, ImageNetDataset
from yolov2.utils.config import load_classes, load_config

cfg = load_config('config/resnet18.yml')
cls_name = load_classes('config/voc_classes.txt')
preprocess = A.Compose([
    A.Resize(416,416),
    A.ColorJitter(),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(),
    A.Normalize(mean=(0,0,0), std=(1,1,1)), #changeable
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.3))

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

class TestVOCDatasetCsv():

    def test_get_img(self):
        trainset = VOC_dataset_csv(csv_root=cfg['DATA']['VOC']['CSV_ROOT'], class_name=cls_name, 
                                mode='trainval')
        
        pick = random.randint(0, len(trainset))
        _, path = trainset.get_label(pick)
        img = trainset.get_img(path)
        
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert img.shape[2] == 3

    def test_get_label(self):
        trainset = VOC_dataset_csv(csv_root=cfg['DATA']['VOC']['CSV_ROOT'], class_name=cls_name, 
                                mode='trainval')
        
        pick = random.randint(0, len(trainset))
        label, _ = trainset.get_label(pick)

        for i in range(6):
            assert all(isinstance(l[i], float) for l in label)

    def test_getitem(self):
        trainset = VOC_dataset_csv(csv_root=cfg['DATA']['VOC']['CSV_ROOT'], class_name=cls_name, 
                                mode='trainval',transform=preprocess)

        pick = random.randint(0, len(trainset))
        img, label = trainset[pick]

        assert_equal(img.shape, torch.Size([3, 416, 416]))
        for i in range(6):
            assert all(isinstance(l[i], float) for l in label)