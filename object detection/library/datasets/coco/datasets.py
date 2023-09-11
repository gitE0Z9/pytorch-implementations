import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
        label_file_path = (
            Path(root)
            .joinpath("annotations")
            .joinpath(f"instances_{mode_filename}{year}.json")
        )

        self.root = root
        self.year = year
        self.mode = mode
        self.class_names = class_names
        self.transform = transform
        self.labels = self.get_labels(label_file_path)

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
            is_bbox = self.transform.to_dict()["transform"].get("bbox_params", False)

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

        annotations = defaultdict(list)
        labeled_image_id = set()
        for annotation in dom["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
            labeled_image_id.add(annotation["image_id"])
        dom["annotations"] = dict(annotations)

        # remove image without label
        # removed_count = 0
        # image_id = set(image["id"] for image in dom["images"])
        # unlabeled_image_id = image_id.difference(labeled_image_id)
        removed_image = [
            image for image in dom["images"] if image["id"] not in labeled_image_id
        ]
        for image in removed_image:
            dom["images"].remove(image)
            # removed_count += 1

        # print(len(image_id) == len(labeled_image_id) + removed_count)

        dom["categories"] = {cat["id"]: cat for cat in dom["categories"]}
        dom["images"].sort(key=lambda x: x["id"])

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


class COCODatasetFromCSV(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        csv_root: str,
        class_names: list[str],
        year: str = "2017",
        mode: str = OperationMode.TEST.value,
        transform=None,
    ):
        mode_filename = self.get_mode(mode)

        self.root = root
        self.mode = mode
        self.year = year
        self.class_names = class_names
        self.transform = transform

        self.table = pd.read_csv(f"{csv_root}/coco_{mode_filename}.csv", index_col="id")
        self.table["class_id"] = self.table["class_id"].astype("int")
        # self.table["anchor_id"] = self.table["anchor_id"].astype("int")

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

    def get_mode(self, mode: str) -> str:
        mapping = {
            OperationMode.TRAIN.value: "train",
            OperationMode.TEST.value: "val",
        }

        return mapping.get(mode, "val")

    def get_img(self, path: str) -> tuple[np.ndarray, int, int]:
        mode_filename = self.get_mode(self.mode)

        img_path = (
            Path(self.root)
            .joinpath(f"{mode_filename}{self.year}")
            .joinpath(path)
            .as_posix()
        )

        img = load_image(img_path)

        return img

    def get_label(self, idx):
        mode_filename = self.get_mode(self.mode)

        img_table = self.table.loc[idx]
        if isinstance(img_table, pd.Series):
            label = [img_table[["cx", "cy", "w", "h", "class_id"]].to_numpy().tolist()]
            # path = img_table["name"].apply(
            #     lambda p: os.path.join(self.root, "VOCdevkit", p)
            # )
            path = img_table["name"]
            path = (
                Path(self.root)
                .joinpath(f"{mode_filename}{self.year}")
                .joinpath(path)
                .as_posix()
            )
        else:
            label = img_table[["cx", "cy", "w", "h", "class_id"]].to_numpy().tolist()
            path = img_table.iloc[0]["name"]
            path = (
                Path(self.root)
                .joinpath(f"{mode_filename}{self.year}")
                .joinpath(path)
                .as_posix()
            )

        return label, path
