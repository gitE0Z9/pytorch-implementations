import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.image import load_image


class ImageDataset(Dataset):
    def __init__(self, pattern: str, transform=None, debug_size: int = 0):
        assert debug_size >= 0, "debug size must be non-negative"

        self.path = list(glob(pattern))
        self.transform = transform
        self.debug_size = debug_size

    def __len__(self) -> int:
        return self.debug_size or len(self.path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pic = load_image(self.path[idx])
        if self.transform:
            pic = self.transform(pic)

        return pic


class Pix2PixDataset(ImageDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        pic = super().__getitem__(idx)

        # a pair of horizontal images
        return pic.split(pic.size(-1) // 2, -1)


class ImagePairDataset(Dataset):
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
            transformed = self.transform(image=content)
            content = transformed["image"]

            transformed = self.transform(image=style)
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

    def get_img(self, filename: str | Path) -> tuple[np.ndarray, int, int]:
        img = load_image(filename, is_numpy=True)
        h, w, _ = img.shape

        return img, h, w
