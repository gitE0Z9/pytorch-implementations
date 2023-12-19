from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchlake.common.utils.image import load_image

LABEL_COLORS = np.array(
    [
        # 0=background
        (0, 0, 0),
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
    ]
)


class VocSegmentation(Dataset):
    def __init__(self, root: str, year: str = "2012", transform=None):
        self.transform = transform
        self.year = year
        self.root = Path(root) / f"VOCdevkit/VOC{self.year}/"

        self.mask_path = self.root / "SegmentationClass"

        self.mask_files = glob(self.mask_path.joinpath("*.png").as_posix())

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx: int):
        image = load_image(
            self.mask_files[idx]
            .replace("png", "jpg")
            .replace("SegmentationClass", "JPEGImages"),
            is_numpy=True,
        )

        mask = Image.open(self.mask_files[idx])
        mask = np.array(mask)
        mask = np.where(mask == 255, 0, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask
