from pathlib import Path
from xml.etree import cElementTree as etree

import numpy as np
import pandas as pd
import torch
from torchlake.common.constants import VOC_CLASS_NAMES
from torchlake.object_detection.constants.enums import OperationMode
from torchlake.object_detection.datasets.schema import DatasetCfg
from torch.utils.data import Dataset

from tqdm import tqdm

from ...utils.image import load_image


class VOCDetectionRaw(Dataset):
    def __init__(
        self,
        root: str | None = None,
        year: str = "2012+2007",
        mode: str = OperationMode.TEST.value,
        class_names: list[str] = [],
        transform=None,
    ):
        # self.config = DatasetCfg(
        #     **load_config(Path(__file__).parent.joinpath("config.yml"))
        # )
        # self.root = Path(root or self.config.ROOT)
        self.root = Path(root)
        self.year = year
        # self.mode = mode
        # self.class_names = class_names or load_classes(
        #     Path(__file__).parent.parent.parent.joinpath(self.config.CLASSES_PATH)
        # )
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
        return Path(__file__).parent.joinpath(f"voc_{self.get_mode_filename()}.csv")

    def get_img(self, idx: int):
        img: np.ndarray = load_image(self.get_img_filename(idx), is_numpy=True)
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


class VOCDetectionFromCSV(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: str,
        # class_names: list[str] = VOC_CLASS_NAMES,
        # mode: str = OperationMode.TEST.value,
        transform=None,
    ):
        # self.config = DatasetCfg(
        #     **load_config(Path(__file__).parent.joinpath("config.yml"))
        # )
        # self.root = Path(root or self.config.ROOT)
        # self.csv_root = Path(csv_path or self.config.CSV_ROOT)
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        # self.mode = mode
        # self.class_names = class_names or load_classes(
        #     Path(__file__).parent.parent.parent.joinpath(self.config.CLASSES_PATH)
        # )
        # self.class_names = class_names
        self.transform = transform

        self.table = pd.read_csv(
            # self.csv_path.joinpath(f"voc_{self.get_mode_filename()}.csv").as_posix(),
            self.csv_path.as_posix(),
            index_col="id",
            dtype={"class_id": pd.Int8Dtype()},
        )

    def __len__(self):
        return self.table.index.nunique()

    def __getitem__(self, idx):
        label, path = self._get_label(idx)
        img = self._get_img(path)

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

    # def _get_mode_filename(self) -> str:
    #     mapping = {
    #         OperationMode.TRAIN.value: "trainval",
    #         OperationMode.TEST.value: "test",
    #     }

    #     return mapping.get(self.mode, "test")

    def _get_img(self, path: str) -> np.ndarray:
        img_path = path.replace("Annotations", "JPEGImages").replace("xml", "jpg")
        img: np.ndarray = load_image(img_path, is_numpy=True)

        return img

    def _get_label(self, idx: int) -> tuple[list, str]:
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
