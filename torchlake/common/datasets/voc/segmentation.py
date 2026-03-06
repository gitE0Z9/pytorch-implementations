from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from ...utils.image import load_image
from .types import YEARS


class VOCSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        year: YEARS = "2012",
        mode: Literal["train", "val", "test", "trainval"] = "val",
        transform=None,
        label_transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.label_transform = label_transform
        self.year = year
        self.mode = mode

        list_path = (
            self.root
            / f"VOC{self.year}"
            / "ImageSets"
            / "Segmentation"
            / f"{self.mode}.txt"
        )
        self.ids = [img_id.strip() for img_id in list_path.read_text().splitlines()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image = load_image(self.get_img_filename(idx), is_numpy=True)

        mask = load_image(self.get_label_filename(idx))
        mask = np.array(mask)
        mask = np.where(mask == 255, 0, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if self.label_transform:
            mask = self.label_transform(mask)

        return image, mask

    def get_img_filename(self, idx: int) -> str:
        return self.root / f"VOC{self.year}" / "JPEGImages" / f"{self.ids[idx]}.jpg"

    def get_label_filename(self, idx: int) -> str:
        return (
            self.root / f"VOC{self.year}" / "SegmentationClass" / f"{self.ids[idx]}.png"
        )
