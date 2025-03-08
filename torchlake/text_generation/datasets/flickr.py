from pathlib import Path

import lmdb
import pandas as pd
import PIL
import PIL.Image
from torch.utils.data import Dataset
from torchlake.common.mixins.dataset import LMDBMixin
from torchlake.common.utils.image import load_image
from tqdm import tqdm


class Flickr8k(Dataset):

    def __init__(
        self,
        img_root: str | Path,
        caption_path: str | Path,
        transform=None,
        text_transform=None,
    ):
        """Flickr 8K caption dataset
        data from [kaggle/adityajn105](https://www.kaggle.com/datasets/adityajn105/flickr8k)

        Args:
            img_root (str | Path): root to image folder
            caption_path (str | Path): path to captions.txt
            transform (torchvision.transforms, optional): image transform. Defaults to None.
            text_transform (torchtext.transforms, optional): text transform. Defaults to None.
        """
        super().__init__()
        self.caption_path = caption_path
        self.img_root = Path(img_root)
        self.data = pd.read_csv(caption_path).to_dict(orient="split")["data"]
        self.transform = transform
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        img_name, caption = self.data[idx]
        img_path = self.img_root.joinpath(img_name)

        image = load_image(img_path, is_numpy=True)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.text_transform is not None:
            caption = self.text_transform(caption)

        return image, caption

    def to_lmdb(self, env: lmdb.Environment):
        with env.begin(write=True) as tx:
            for i, (img, labels) in enumerate(tqdm(self)):
                if i % 5 == 0:
                    tx.put(f"{i // 5}".encode("utf-8"), img.tobytes())
                    tx.put(
                        f"{i // 5}_shape".encode("utf-8"),
                        str(list(img.shape)).encode("utf-8"),
                    )

                tx.put(f"{i}_label".encode("utf-8"), str(labels).encode("utf-8"))

            tx.put(b"count", str(len(self)).encode("utf-8"))


class Flickr8kFromLMDB(LMDBMixin, Dataset):

    def __init__(
        self,
        lmdb_path: str | Path,
        transform=None,
        text_transform=None,
    ):
        """Flickr 8K caption dataset
        data from [kaggle/adityajn105](https://www.kaggle.com/datasets/adityajn105/flickr8k)

        Args:
            lmdb_path (str | Path): root to lmdb folder
            transform (torchvision.transforms, optional): image transform. Defaults to None.
            text_transform (torchtext.transforms, optional): text transform. Defaults to None.
        """
        super().__init__()
        self.transform = transform
        self.text_transform = text_transform

        self.post_init(lmdb_path)

    def __getitem__(self, idx: int) -> tuple:
        if idx >= self.data_size:
            raise IndexError(f"invalid index {idx}")

        image = self.get_image(idx // 5)
        caption = self.get_label(idx)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.text_transform is not None:
            caption = self.text_transform(caption)

        return image, caption

    def get_label(self, idx: int) -> list[list[list[int, int, int]]]:
        with self.env.begin() as tx:
            label = tx.get(f"{idx}_label".encode("utf-8")).decode()

        return label
