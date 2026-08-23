import string
import uuid
from base64 import b64decode, b64encode
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchtext.vocab import Vocab
from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.text import build_vocab


class SyntheticTextDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        vocab: Vocab | None = None,
        ignored: str = "😀",
        transform=None,
    ):
        self.root = Path(root)
        self.ignored = ignored

        if vocab is None:
            vocab = build_vocab(
                string.printable,
                NLPContext(min_frequency=1, special_tokens=[]),
            )
            vocab.insert_token(self.ignored, 0)
            vocab.set_default_index(0)

        self.token_table = vocab

        self.images = list(self.root.glob("*.png"))
        self.transform = transform
        self.generator = None

        self.label_path = self.root.joinpath("label.txt")
        self.labels = self.build_labels()

    def set_generator(self, generator):
        self.generator = generator

    def build_labels(self) -> dict[str, str]:
        if self.label_path.exists():
            rows = self.label_path.read_text().split("\n")
            return {row.split(",")[0]: row.split(",")[1] for row in rows if row}

    def build_images(self, size: int = 1000):
        assert self.generator is not None, "please set_generator first."

        count = 0
        while count < size:
            for img, label in tqdm(self.generator):
                # is label oov
                if any(l not in self.token_table for l in label):
                    continue

                img_filename = f"{uuid.uuid4().hex}.png"
                img_path = self.root.joinpath(img_filename)
                img.save(img_path)
                self.label_path.open("a").write(
                    f"{img_filename},{b64encode(label.encode()).decode()}\n"
                )
                count += 1
                if count >= size:
                    break

        self.images = list(self.root.glob("*.png"))
        self.labels = self.build_labels()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = b64decode(self.labels[img_path.stem + ".png"]).decode()

        image = self.transform(image)
        label = torch.Tensor([self.token_table[s] for s in label]).long()
        return image, label
