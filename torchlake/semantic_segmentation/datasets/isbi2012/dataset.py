from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from torchlake.common.utils.image import load_image


class ISBI2012(Dataset):
    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "test"] = "test",
        transforms: (
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
            | None
        ) = None,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.mode = mode

        self.img_folder = self.root / mode / "imgs"
        self.label_folder = self.root / mode / "labels"
        self.ids = [img_path.stem for img_path in self.img_folder.glob("*.png")]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image = load_image(self.get_img_filename(idx), is_numpy=True)

        mask = load_image(self.get_label_filename(idx))
        mask = np.array(mask)
        mask = np.where(mask == 255, 0, 1)

        if self.transforms is not None:
            return self.transforms(image, mask)

        return image, mask

    def get_img_filename(self, idx: int) -> str:
        return self.img_folder / f"{self.ids[idx]}.png"

    def get_label_filename(self, idx: int) -> str:
        return self.label_folder / f"{self.ids[idx]}.png"
