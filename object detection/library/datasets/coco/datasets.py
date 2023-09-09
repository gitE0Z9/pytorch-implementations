import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from constants.enums import OperationMode
from utils.plot import load_image


class COCODatasetRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        class_names: list[str],
        year: str = "2017",
        mode: str = OperationMode.TEST.value,
        transform=None,
    ):
        mode_filename = self.get_mode(mode)

        self.root = root
        self.year = year
        self.mode = mode
        self.class_names = class_names
        self.transform = transform
        self.labels = self.get_labels(
            Path(self.root).joinpath(
                f"annotations/instances_{mode_filename}{self.year}.json"
            )
        )

    def __len__(self) -> int:
        """how many pictures in the dataset

        Returns:
            int: picture count
        """
        return len(self.labels["images"])

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

    def get_mode(self, mode: str) -> str:
        """get mode representation in filename

        Args:
            mode (str): train or test

        Returns:
            str: mode representation in filename
        """
        mapping = {
            OperationMode.TRAIN.value: "train",
            OperationMode.TEST.value: "val",
        }

        return mapping.get(mode, "val")

    def get_labels(self, annotation_path: str):
        with open(annotation_path, "r") as f:
            dom = json.loads(f.read())

        dom["images"].sort(key=lambda x: x["id"])

        annotations = defaultdict(list)
        for annotation in dom["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        dom["annotations"] = annotations

        dom["categories"] = {cat["id"]: cat for cat in dom["categories"]}

        return dom

    def get_img(self, idx: int) -> tuple[np.ndarray, int, int]:
        mode_filename = self.get_mode(self.mode)

        img_path = (
            Path(self.root)
            .joinpath(f"{mode_filename}{self.year}")
            .joinpath(self.labels["images"][idx]["file_name"])
            .as_posix()
        )

        img = load_image(img_path)
        h, w, _ = img.shape

        return img, h, w

    def get_label(self, idx: int, h: int, w: int) -> list:
        img_id = self.labels["images"][idx]["id"]
        label = self.labels["annotations"][img_id]
        label = self.process_label(label, h, w)

        return label

    def process_label(self, labels: list, img_h: int, img_w: int) -> list:
        new_labels = []

        for label in labels:
            if label["iscrowd"] != 0:
                continue
            bbox = label["bbox"]

            xmin = float(bbox[0])
            xmax = float(bbox[0] + bbox[2])
            ymin = float(bbox[1])
            ymax = float(bbox[1] + bbox[3])

            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            class_name = self.labels["categories"][label["category_id"]]["name"]
            class_idx = self.class_names.index(class_name)

            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h

            new_labels.append([cx, cy, w, h, class_idx])

        return new_labels
