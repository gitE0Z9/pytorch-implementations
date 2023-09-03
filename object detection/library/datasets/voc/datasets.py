import os
from typing import List
from xml.etree import cElementTree as etree

import pandas as pd
import torch
from utils.plot import load_image


class VOCDatasetRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        class_name: List[str],
        year: str = "2012+2007",
        mode: str = "trainval",
        transform=None,
    ):
        list_path = [
            (os.path.join(root, f"VOCdevkit/VOC{y}/ImageSets/Main/{mode}.txt"), y)
            for y in year.split("+")
        ]

        self.root = root
        self.year = year
        self.mode = mode
        self.class_name = class_name
        self.transform = transform
        self.labels = []
        for p, y in list_path:
            with open(p, "r") as f:
                for g in f.readlines():
                    self.labels.append(
                        os.path.join(
                            root, f"VOCdevkit/VOC{y}/Annotations", f"{g.strip()}.xml"
                        )
                    )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, h, w = self.get_img(idx)
        label = self.get_label(idx, h, w)

        if self.transform:
            is_bbox = self.transform.to_dict()["transform"]["bbox_params"]

            kwargs = dict(image=img)

            if is_bbox:
                kwargs["bboxes"] = label

            transformed = self.transform(**kwargs)
            img = transformed["image"]

            if is_bbox:
                label = transformed["bboxes"]

        return img, label

    def get_img(self, idx):
        jpg = (
            self.labels[idx].replace("Annotations", "JPEGImages").replace("xml", "jpg")
        )

        img = load_image(jpg)
        h, w, _ = img.shape

        return img, h, w

    def get_label(self, idx, h, w):
        xml = self.labels[idx]
        tree = etree.parse(xml)
        label = self.process_label(tree, h, w)

        return label

    def process_label(self, tree, img_h: int, img_w: int) -> list:
        tree = tree.findall("object")

        label = []
        for obj in tree:
            if obj.find("difficult").text == "1":
                continue
            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            xmax = float(bbox.find("xmax").text)
            ymin = float(bbox.find("ymin").text)
            ymax = float(bbox.find("ymax").text)

            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            class_ind = self.class_name.index(obj.find("name").text)

            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h

            label.append([cx, cy, w, h, class_ind])

        return label


class VOCDatasetFromCSV(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        csv_root: str,
        class_name: List[str],
        mode: str = "trainval",
        transform=None,
    ):
        self.root = root
        self.mode = mode
        self.class_name = class_name
        self.transform = transform

        self.table = pd.read_csv(f"{csv_root}/voc_{mode}.csv", index_col="id")
        self.table["class_id"] = self.table["class_id"].astype("int")
        self.table["anchor_id"] = self.table["anchor_id"].astype("int")

    def __len__(self):
        return self.table.index.nunique()

    def __getitem__(self, idx):
        label, path = self.get_label(idx)
        img = self.get_img(path)

        if self.transform:
            is_bbox = self.transform.to_dict()["transform"]["bbox_params"]

            kwargs = dict(image=img)

            if is_bbox:
                kwargs["bboxes"] = label

            transformed = self.transform(**kwargs)
            img = transformed["image"]

            if is_bbox:
                label = transformed["bboxes"]

        return img, label

    def get_img(self, path):
        jpg = path.replace("Annotations", "JPEGImages").replace("xml", "jpg")

        img = load_image(jpg)

        return img

    def get_label(self, idx):
        img_table = self.table.loc[idx]
        if isinstance(img_table, pd.Series):
            label = [
                img_table[["cx", "cy", "w", "h", "class_id", "anchor_id"]]
                .to_numpy()
                .tolist()
            ]
            path = img_table["name"].apply(lambda p: os.path.join(self.root, p))
        else:
            label = (
                img_table[["cx", "cy", "w", "h", "class_id", "anchor_id"]]
                .to_numpy()
                .tolist()
            )
            path = os.path.join(self.root, img_table.iloc[0]["name"])

        return label, path
