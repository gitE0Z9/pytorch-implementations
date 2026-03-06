import json
from pathlib import Path
from typing import Literal
from xml.etree import cElementTree as etree

import lmdb
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm

from torchlake.common.constants import VOC_CLASS_NAMES

from ...utils.image import load_image
from .types import YEARS


class VOCDetectionRaw(Dataset):
    def __init__(
        self,
        root: str,
        mode: Literal["trainval", "test"],
        years: list[YEARS] = ["2007", "2012"],
        class_names: list[str] = VOC_CLASS_NAMES,
        transform=None,
    ):
        self.root = Path(root)
        self.years = years
        self.mode = mode
        self.class_names = class_names
        self.transform = transform

        # each xml file
        self.ids: list[Path] = []
        for year in self.years:
            dir = self.root.joinpath(f"VOC{year}")
            list_path = (
                dir.joinpath("ImageSets").joinpath("Main").joinpath(f"{self.mode}.txt")
            )

            for img_id in list_path.read_text().splitlines():
                self.ids.append(
                    dir.joinpath("Annotations").joinpath(f"{img_id.strip()}.xml")
                )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(f"invalid index {idx}")

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

    def get_img_filename(self, idx: int) -> str:
        return self.ids[idx].replace("Annotations", "JPEGImages").replace("xml", "jpg")

    def get_img(self, idx: int) -> tuple[np.ndarray, int, int]:
        img: np.ndarray = load_image(self.get_img_filename(idx), is_numpy=True)
        h, w, _ = img.shape

        return img, h, w

    def get_label(self, idx: int, h: int, w: int) -> list[list[float]]:
        xml = self.ids[idx]
        tree = etree.parse(xml)

        return self.process_label(tree, h, w)

    def process_label(
        self,
        tree: etree.ElementTree,
        img_h: int,
        img_w: int,
    ) -> list[list[float]]:
        objs = tree.findall("object")

        label = []
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

            class_ind = self.class_names.index(obj.find("name").text)

            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h

            label.append([cx, cy, w, h, class_ind])

        return label

    def to_polars(self) -> pl.DataFrame:
        data = []
        for idx, (_, labels) in enumerate(tqdm(self)):
            img_filename = self.get_img_filename(idx)
            for label in labels:
                placeholder = [idx, img_filename]
                placeholder.extend(label)
                data.append(placeholder)

        return pl.DataFrame(
            data,
            schema=["id", "name", "cx", "cy", "w", "h", "class_id"],
        )

    def to_csv(self, csv_path: str):
        df: pl.DataFrame = self.to_polars()
        df.write_csv(csv_path)

    def to_lmdb(self, env: lmdb.Environment):
        with env.begin(write=True) as tx:
            for i, (img, labels) in enumerate(tqdm(self)):
                tx.put(f"{i}".encode("utf-8"), img.tobytes())
                tx.put(
                    f"{i}_shape".encode("utf-8"), str(list(img.shape)).encode("utf-8")
                )
                tx.put(f"{i}_label".encode("utf-8"), str(labels).encode("utf-8"))

            tx.put(b"count", str(len(self)).encode("utf-8"))


class VOCDetectionFromCSV(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: str,
        transform=None,
    ):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.transform = transform

        df: pl.DataFrame = pl.read_csv(
            self.csv_path.as_posix(),
            schema_overrides={"class_id": pl.Int8},
        )

        grouped = df.group_by("id", maintain_order=True).agg(
            [
                pl.col("name").first(),
                pl.struct(["cx", "cy", "w", "h", "class_id"]).alias("labels"),
            ]
        )

        self.labels = grouped["labels"].to_list()
        self.paths = grouped["name"].to_list()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        label, path = self._get_label(idx)
        img = self._get_img(path)

        if self.transform is not None:
            is_bbox = self.transform.to_dict()["transform"]["bbox_params"]

            kwargs = dict(image=img)

            if is_bbox:
                kwargs["bboxes"] = label

            transformed = self.transform(**kwargs)
            img = transformed["image"]

            if is_bbox:
                label = transformed["bboxes"]

        return img, label

    def _get_img(self, path: str) -> np.ndarray:
        img_path = path.replace("Annotations", "JPEGImages").replace("xml", "jpg")
        img: np.ndarray = load_image(img_path, is_numpy=True)

        return img

    def _get_label(self, idx: int) -> tuple[list, str]:
        label = self.labels[idx]
        label = [
            [bbox["cx"], bbox["cy"], bbox["w"], bbox["h"], bbox["class_id"]]
            for bbox in label
        ]
        path = self.root.joinpath(self.paths[idx]).as_posix()

        return label, path


class VOCDetectionFromLMDB(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        transform=None,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.transform = transform

        self.env = lmdb.open(lmdb_path)

        self.data_size = self.get_data_size()

    def get_data_size(self) -> int:
        with self.env.begin() as tx:
            return int(tx.get(b"count"))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if idx >= self.data_size:
            raise IndexError(f"invalid index {idx}")

        ori_img = self._get_img(idx)
        ori_label = self._get_label(idx)

        if self.transform is None:
            return ori_img, ori_label

        for _ in range(5):
            new_img, new_label = self._transform(ori_img, ori_label)
            if len(new_label) > 0:
                return new_img, new_label

        raise ValueError

    def _get_img(self, idx: int) -> np.ndarray:
        with self.env.begin() as tx:
            shape = json.loads(tx.get(f"{idx}_shape".encode("utf-8")))
            img: np.ndarray = np.frombuffer(
                tx.get(f"{idx}".encode("utf-8")), np.uint8
            ).reshape(shape)

        return img

    def _get_label(self, idx: int) -> list[list[int]]:
        with self.env.begin() as tx:
            label = json.loads(tx.get(f"{idx}_label".encode("utf-8")))

        return label

    def _transform(
        self,
        img: np.ndarray,
        label: list[list[int]],
    ) -> tuple[np.ndarray, list[list[int]]]:
        is_bbox = self.transform.to_dict()["transform"]["bbox_params"]

        kwargs = dict(image=img)

        if is_bbox:
            kwargs["bboxes"] = label

        transformed = self.transform(**kwargs)
        img = transformed["image"]

        if is_bbox:
            label = transformed["bboxes"]

        return img, label
