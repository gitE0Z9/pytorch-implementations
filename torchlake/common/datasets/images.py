import random
from functools import lru_cache
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DatasetBaseMixin:
    def __init__(self, pattern: str, transform=None, debug_size: int = 0):
        self.path = list(glob(pattern))
        self.transform = transform
        self.debug_size = debug_size

    def __len__(self) -> int:
        return len(self.path) if self.debug_size <= 0 else self.debug_size


class ImageDataset(DatasetBaseMixin, Dataset):
    def __getitem__(self, idx: int) -> torch.Tensor:
        pic = Image.open(self.path[idx]).convert("RGB")
        if self.transform:
            pic = self.transform(pic)

        return pic


class Pix2PixDataset(DatasetBaseMixin, Dataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        pic = Image.open(self.path[idx]).convert("RGB")

        if self.transform:
            pic = self.transform(pic)

        return pic.split(pic.size(-1) // 2, -1)


class ImagePairDataset(DatasetBaseMixin, Dataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        pic = Image.open(self.path[idx]).convert("RGB")
        if self.transform:
            pic = self.transform(pic)

        return pic


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, content_root: str, style_root: str, transform=None):
        self.content_root = Path(content_root)
        self.style_root = Path(style_root)
        self.transform = transform
        self.contents = list(self.content_root.glob("**/*.jpg"))
        self.styles = list(self.style_root.glob("*"))

    def __len__(self) -> int:
        # return 30000
        return len(self.contents)  # * len(self.styles)

    def __getitem__(self, idx: int) -> tuple[np.ndarray]:
        content, _, _ = self.get_content_img(idx)
        style, _, _ = self.get_style_img()

        if self.transform:
            kwargs = dict(image=content)
            transformed = self.transform(**kwargs)
            content = transformed["image"]

            kwargs = dict(image=style)
            transformed = self.transform(**kwargs)
            style = transformed["image"]

        return content, style

    def get_content_img(self, idx: int) -> tuple[np.ndarray, int, int]:
        collection = self.contents
        idx = idx % len(self.contents)

        return self.get_img(collection[idx].as_posix())

    def get_style_img(self) -> tuple[np.ndarray, int, int]:
        collection = self.styles
        idx = random.randint(0, len(self.styles) - 1)  # idx // len(self.contents)

        # if not is_content: print(idx)

        return self.get_img(collection[idx].as_posix())

    @lru_cache
    def get_img(self, filename: str) -> tuple[np.ndarray, int, int]:
        img = cv2.imread(filename)[:, :, ::-1]
        h, w, _ = img.shape

        return img, h, w
