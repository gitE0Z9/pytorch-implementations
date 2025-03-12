from pathlib import Path
from typing import Literal
from xml.etree import cElementTree as etree

from torch.utils.data import Dataset
from torchlake.common.datasets.imagenet import (
    IMAGENET_CLASS_NOS,
    IMAGENET_DARKNET_CLASS_NOS,
)
from torchlake.common.utils.image import load_image


class ImageNetFromXML(Dataset):
    def __init__(
        self,
        root: str | Path,
        mode: Literal["val"] = "val",
        label_encode: Literal["imagenet", "darknet"] = "imagenet",
        max_class: int = 1000,
        transform=None,
    ):
        self.root = Path(root)
        self.labels = {
            "imagenet": IMAGENET_CLASS_NOS,
            "darknet": IMAGENET_DARKNET_CLASS_NOS,
        }[label_encode][:max_class]
        self.transform = transform
        self.max_class = max_class

        img_root = self.root.joinpath("Data").joinpath("CLS-LOC").joinpath(mode)
        self.imgs = list(img_root.glob("*"))

        ann_root = self.root.joinpath("Annotations").joinpath("CLS-LOC").joinpath(mode)
        self.anns = list(ann_root.glob("*"))

        assert len(self.imgs) == len(
            self.anns
        ), "number of image and number of annotation misaligned."

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = load_image(self.imgs[idx], is_numpy=True)

        if self.transform:
            img = self.transform(image=img)["image"]

        tree = etree.parse(self.anns[idx])
        tree = tree.find("object").find("name").text
        label = self.labels.index(tree)

        return img, label
