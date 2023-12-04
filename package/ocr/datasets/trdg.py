import string
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from trdg.generators import GeneratorFromWikipedia
import uuid
from base64 import b64encode, b64decode

class SyntheticTextDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.ignored = "😀"

        self.images = list(self.root.glob("*.png"))
        self.transform = transform

        self.label_path = self.root.joinpath("label.txt")
        self.token_table = self.build_label_table()
        self.labels = self.build_labels()
        
    def build_labels(self) -> dict[str, str]:
        if self.label_path.exists():
            rows = self.label_path.read_text().split("\n")
            return {row.split(',')[0]: row.split(',')[1] for row in rows if row}

    def build_images(self, size: int = 1000):
        generator = GeneratorFromWikipedia()
        count = 0
        while count < size:
            for img, label in generator:
                is_label_oov = any(l not in self.token_table for l in label)
                if is_label_oov: continue
                img_filename = f"{uuid.uuid4().hex}.png"
                img_path = self.root.joinpath(img_filename)
                img.save(img_path)
                self.label_path.open('a').write(f'{img_filename},{b64encode(label.encode()).decode()}\n')
                count += 1
                if count >= size: break

        self.images = list(self.root.glob("*.png"))
        self.labels = self.build_labels()

    def build_label_table(self):
        ascii_table = {s: i + 1 for i, s in enumerate(string.printable)}
        ascii_table[self.ignored] = 0

        return ascii_table

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = b64decode(self.labels[img_path.stem + ".png"]).decode()

        image = self.transform(image)
        label = torch.Tensor([self.token_table.get(s, 0) for s in label]).long()
        return image, label
