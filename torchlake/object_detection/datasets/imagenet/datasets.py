import os
from pathlib import Path
from xml.etree import cElementTree as etree

import torch
from torchlake.common.utils.image import load_image
from torchlake.object_detection.constants.enums import OperationMode
from torchlake.object_detection.datasets.schema import DatasetCfg
from torchlake.object_detection.utils.config import load_classes, load_config


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | None = None,
        mode: str = OperationMode.TEST.value,
        class_names: list[str] = [],
        transform=None,
    ):
        self.config = DatasetCfg(**load_config("datasets/imagenet/config.yml"))
        self.root = Path(root or self.config.ROOT)
        self.mode = mode
        self.class_names = class_names or load_classes(self.config.CLASSES_PATH)
        self.transform = transform

        if self.mode == OperationMode.TRAIN.value:
            img_pattern = (
                self.root.joinpath("Data")
                .joinpath("CLS-LOC")
                .joinpath(self.get_mode_filename())
                .glob("*/*")
            )
            self.img_list = list(img_pattern)

        elif self.mode == OperationMode.TEST.value:
            ann_dir = (
                self.root.joinpath("Annotations")
                .joinpath("CLS-LOC")
                .joinpath(self.get_mode_filename())
                .glob("*.xml")
            )
            self.ann_list = list(ann_dir)
            img_dir = (
                self.root.joinpath("Data")
                .joinpath("CLS-LOC")
                .joinpath(self.get_mode_filename())
                .glob("*")
            )
            self.img_list = list(img_dir)

            assert len(self.ann_list) == len(self.img_list), "misalign"

        else:
            raise ValueError

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img_path = self.img_list[idx]
        img = load_image(img_path, is_numpy=True)

        if self.transform:
            img = self.transform(image=img)["image"]

        if self.mode == OperationMode.TRAIN.value:
            label = os.path.basename(img_path).split("_")[0]
            label = self.class_names.index(label)
        elif self.mode == OperationMode.TEST.value:
            tree = etree.parse(self.ann_list[idx])
            tree = tree.find("object").find("name").text
            label = self.class_names.index(tree)

        return img, label

    def get_mode_filename(self) -> str:
        mapping = {
            OperationMode.TRAIN.value: "train",
            OperationMode.TEST.value: "val",
        }

        return mapping.get(self.mode, "val")
