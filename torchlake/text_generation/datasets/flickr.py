from pathlib import Path

import pandas as pd
import PIL
import PIL.Image
from torch.utils.data import Dataset
from torchlake.common.utils.image import load_image


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

        img: PIL.Image = load_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.text_transform is not None:
            caption = self.text_transform(caption)

        return img, caption
