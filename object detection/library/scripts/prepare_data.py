import os
from xml.etree import cElementTree as etree

import pandas as pd
import torch
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from utils.config import load_anchors, load_classes

dataset_kind = [("2012", "trainval"), ("2007", "trainval"), ("2007", "test")]

label_path = {"trainval": [], "test": []}

labels = {"trainval": [], "test": []}

class_name = load_classes("config/voc_classes.txt")
anchors = load_anchors("config/anchors.txt").view(-1, 2)

if __name__ == "__main__":
    for year, mode in dataset_kind:
        root = "../data"  #'D://research/pytorch implementation/data'
        name_file_path = os.path.join(
            root, f"VOCdevkit/VOC{year}/ImageSets/Main/{mode}.txt"
        )
        with open(name_file_path, "r") as f:
            for g in f.readlines():
                annot_path = os.path.join(
                    root, f"VOCdevkit/VOC{year}/Annotations", f"{g.strip()}.xml"
                )
                label_path[mode].append(annot_path)

    for mode, annot_path in label_path.items():
        for index, ann in tqdm(enumerate(annot_path)):
            tree = etree.parse(ann)
            img_w = float(tree.find("size").find("width").text)
            img_h = float(tree.find("size").find("height").text)
            objs = tree.findall("object")
            for obj in objs:
                if obj.find("difficult").text == "1":
                    continue
                bbox = obj.find("bndbox")

                xmin = float(bbox.find("xmin").text)
                xmax = float(bbox.find("xmax").text)
                ymin = float(bbox.find("ymin").text)
                ymax = float(bbox.find("ymax").text)

                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                class_index = class_name.index(obj.find("name").text)

                cx = (xmin + xmax) / 2 / img_w
                cy = (ymin + ymax) / 2 / img_h

                # slow and repetitive computation, but fine since one-time
                truths = torch.Tensor([[0.5, 0.5, w, h]])
                truths = box_convert(truths, "cxcywh", "xyxy")
                priors = torch.cat([torch.full(anchors.shape, 0.5), anchors], 1)
                priors = box_convert(priors, "cxcywh", "xyxy")
                anchor_index = box_iou(truths, priors).argmax(1).item()

                labels[mode].append(
                    [index, ann, cx, cy, w, h, class_index, anchor_index]
                )

        df = pd.DataFrame(labels[mode])
        df.columns = ["id", "name", "cx", "cy", "w", "h", "class_id", "anchor_id"]
        df.to_csv(f"voc_{mode}.csv", index=False)
