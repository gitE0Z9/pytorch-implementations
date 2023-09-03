import os
from glob import glob
from xml.etree import cElementTree as etree

import torch
from utils.plot import load_image


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, classname: list[str], mode: str, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.classes_name = classname

        if mode == "val":
            ann_dir = os.path.join(
                self.root, "Annotations/CLS-LOC/", self.mode, "*.xml"
            )
            self.ann_list = glob(ann_dir)
            img_dir = os.path.join(self.root, "Data/CLS-LOC/", self.mode, "*")
            self.img_list = glob(img_dir)

            assert len(self.ann_list) == len(self.img_list), "misalign"

        elif mode == "train":
            img_pattern = os.path.join(self.root, "Data/CLS-LOC/", self.mode, "*", "*")
            self.img_list = glob(img_pattern)

        else:
            raise ValueError

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = load_image(img_path)

        if self.transform:
            img = self.transform(image=img)["image"]

        if self.mode == "val":
            tree = etree.parse(self.ann_list[idx])
            tree = tree.find("object").find("name").text
            label = self.classes_name.index(tree)
        elif self.mode == "train":
            label = os.path.basename(img_path).split("_")[0]
            label = self.classes_name.index(label)

        return img, label
