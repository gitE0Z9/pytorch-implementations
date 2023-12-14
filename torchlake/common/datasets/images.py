from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, pattern: str, transform=None):
        self.path = list(glob(pattern))
        self.transform = transform

    def __len__(self):
        # return 1000 # for debug forward
        return len(self.path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pic = Image.open(self.path[idx]).convert("RGB")
        if self.transform:
            pic = self.transform(pic)

        return pic
