from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ...utils.image import load_image
from .types import YEARS


class VOCSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        year: YEARS = "2012",
        transform=None,
        label_transform=None,
    ):
        self.transform = transform
        self.label_transform = label_transform
        self.year = year
        self.root = Path(root).joinpath("VOCdevkit").joinpath(f"VOC{self.year}")

        self.label_root = self.root / "SegmentationClass"
        self.label_files = glob(self.label_root.joinpath("*.png").as_posix())

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx: int):
        image = load_image(
            self.label_files[idx]
            .replace("png", "jpg")
            .replace("SegmentationClass", "JPEGImages"),
            is_numpy=True,
        )

        mask = Image.open(self.label_files[idx])
        mask = np.array(mask)
        mask = np.where(mask == 255, 0, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        if self.label_transform:
            mask = self.label_transform(mask)

        return image, mask
