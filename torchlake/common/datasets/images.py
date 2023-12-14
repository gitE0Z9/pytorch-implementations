from glob import glob

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

        return pic.split(2, -1)


class ImagePairDataset(DatasetBaseMixin, Dataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        pic = Image.open(self.path[idx]).convert("RGB")
        if self.transform:
            pic = self.transform(pic)

        return pic
