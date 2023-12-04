from pathlib import Path
from xml.etree import cElementTree as etree

import pandas as pd
import numpy as np
import torch
from object_detection.utils.plot import load_image
from tqdm import tqdm
from object_detection.utils.config import load_config, load_classes
from object_detection.constants.enums import OperationMode
from object_detection.datasets.schema import DatasetCfg


class VOCDatasetRaw(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | None = None,
        year: str = "2012+2007",
        mode: str = OperationMode.TEST.value,
        class_names: list[str] = [],
        transform=None,
    ):
        self.config = DatasetCfg(**load_config("datasets/voc/config.yml"))
        self.root = Path(root or self.config.ROOT)
        self.year = year
        self.mode = mode
        self.class_names = class_names or load_classes(self.config.CLASSES_PATH)
        self.transform = transform

        self.labels = []
        for y in year.split("+"):
            year_directory = self.root.joinpath(f"VOC{y}")
            list_path = (
                year_directory.joinpath("ImageSets")
                .joinpath("Main")
                .joinpath(f"{self.get_mode_filename()}.txt")
                .as_posix()
            )
            with open(list_path, "r") as f:
                for line in f.readlines():
                    self.labels.append(
                        (
                            year_directory.joinpath("Annotations")
                            .joinpath(f"{line.strip()}.xml")
                            .as_posix()
                        )
                    )

    def __len__(self) -> int:
        return len(self.labels)

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
        mapping = {
            OperationMode.TRAIN.value: "trainval",
            OperationMode.TEST.value: "test",
        }

        return mapping.get(self.mode, "test")

    def get_img_filename(self, idx: int) -> str:
        return (
            self.labels[idx].replace("Annotations", "JPEGImages").replace("xml", "jpg")
        )

    def get_csv_path(self):
        return f"datasets/voc/voc_{self.get_mode_filename()}.csv"

    def get_img(self, idx: int):
        img = load_image(self.get_img_filename(idx))
        h, w, _ = img.shape

        return img, h, w

    def get_label(self, idx: int, h: int, w: int):
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

            class_ind = self.class_names.index(obj.find("name").text)

            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h

            label.append([cx, cy, w, h, class_ind])

        return label

    def to_csv(self, csv_path: str = ""):
        csv_path = csv_path or self.get_csv_path()
        data = []
        for idx, (_, labels) in enumerate(tqdm(self)):
            img_filename = self.get_img_filename(idx)
            for label in labels:
                placeholder = [idx, img_filename]
                placeholder.extend(label)
                data.append(placeholder)

        df = pd.DataFrame(
            data,
            columns=["id", "name", "cx", "cy", "w", "h", "class_id"],
        )
        df.to_csv(csv_path, index=False)


class VOCDatasetFromCSV(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | None = None,
        csv_root: str | None = None,
        mode: str = OperationMode.TEST.value,
        class_names: list[str] = [],
        transform=None,
    ):
        self.config = DatasetCfg(**load_config("datasets/voc/config.yml"))
        self.root = Path(root or self.config.ROOT)
        self.csv_root = Path(csv_root or self.config.CSV_ROOT)
        self.mode = mode
        self.class_names = class_names or load_classes(self.config.CLASSES_PATH)
        self.transform = transform

        self.table = pd.read_csv(
            self.csv_root.joinpath(f"voc_{self.get_mode_filename()}.csv").as_posix(),
            index_col="id",
            dtype={"class_id": "Int8"},
        )

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

    def get_mode_filename(self) -> str:
        mapping = {
            OperationMode.TRAIN.value: "trainval",
            OperationMode.TEST.value: "test",
        }

        return mapping.get(self.mode, "test")

    def get_img(self, path: str) -> np.ndarray:
        img_path = path.replace("Annotations", "JPEGImages").replace("xml", "jpg")
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

        path = self.root.joinpath(path).as_posix()

        return label, path
