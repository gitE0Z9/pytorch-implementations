import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from constants.enums import OperationMode
from utils.plot import load_image
from datasets.schema import DatasetCfg
from utils.config import load_config, load_classes


class COCODatasetRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | None = None,
        year: str = "2017",
        mode: str = OperationMode.TEST.value,
        class_names: list[str] = [],
        transform=None,
    ):
        self.config = DatasetCfg(**load_config("datasets/coco/config.yml"))
        self.root = Path(root or self.config.ROOT)
        self.year = year
        self.mode = mode
        self.class_names = class_names or load_classes(self.config.CLASSES_PATH)
        self.transform = transform

        label_file_path = self.root.joinpath("annotations").joinpath(
            f"instances_{self.get_mode_filename()}{year}.json"
        )
        self.labels = self.get_labels(label_file_path)

    def __len__(self) -> int:
        """how many pictures in the dataset

        Returns:
            int: picture count
        """
        return len(self.labels["images"])

    def __getitem__(self, idx: int):
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

    def get_mode_filename(self) -> str:
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

        return mapping.get(self.mode, "val")

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
        removed_image = filter(
            lambda image: image["id"] not in labeled_image_id,
            dom["images"],
        )
        for image in removed_image:
            dom["images"].remove(image)
            # removed_count += 1

        # print(len(image_id) == len(labeled_image_id) + removed_count)

        dom["categories"] = {cat["id"]: cat for cat in dom["categories"]}
        dom["images"].sort(key=lambda x: x["id"])

        return dom

    def get_img(self, idx: int) -> tuple[np.ndarray, int, int]:
        img_path = (
            Path(self.root)
            .joinpath(f"{self.get_mode_filename()}{self.year}")
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
        root: str | None = None,
        csv_root: str | None = None,
        year: str = "2017",
        mode: str = OperationMode.TEST.value,
        class_names: list[str] = [],
        transform=None,
    ):
        self.config = DatasetCfg(**load_config("datasets/coco/config.yml"))
        self.root = Path(root or self.config.ROOT)
        self.csv_root = Path(csv_root or self.config.CSV_ROOT)
        self.year = year
        self.mode = mode
        self.class_names = class_names or load_classes(self.config.CLASSES_PATH)
        self.transform = transform

        self.table = pd.read_csv(
            self.csv_root.joinpath(f"coco_{self.get_mode_filename()}.csv").as_posix(),
            index_col="id",
            dtype={"class_id": "Int8"},
        )

    def __len__(self) -> int:
        return self.table.index.nunique()

    def __getitem__(self, idx: int):
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

    def get_mode_filename(self) -> str:
        mapping = {
            OperationMode.TRAIN.value: "train",
            OperationMode.TEST.value: "val",
        }

        return mapping.get(self.mode, "val")

    def get_img(self, path: str) -> np.ndarray:
        img_path = (
            Path(self.root)
            .joinpath(f"{self.get_mode_filename()}{self.year}")
            .joinpath(path)
            .as_posix()
        )

        img = load_image(img_path)

        return img

    def get_label(self, idx: int) -> tuple[list, str]:
        img_table = self.table.loc[idx]
        label = img_table[["cx", "cy", "w", "h", "class_id"]].to_numpy().tolist()

        if isinstance(img_table, pd.Series):
            label = [label]
            path = img_table["name"]
            # path = img_table["name"].apply(
            #     lambda p: os.path.join(self.root, "VOCdevkit", p)
            # )
        else:
            path = img_table.iloc[0]["name"]

        path = (
            self.root.joinpath(f"{self.get_mode_filename()}{self.year}")
            .joinpath(path)
            .as_posix()
        )

        return label, path
