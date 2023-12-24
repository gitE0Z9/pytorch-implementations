from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from torchlake.common.utils.image import load_image


class OmniglotSet(Dataset):
    def __init__(
        self,
        root: str,
        set_size: int,
        transform=None,
        language: str | None = None,
        enable_kshot: bool = False,
        shot_size: int = 5,
    ):
        self.root = Path(root) / "omniglot-py" / "images_background"

        self.language_paths = list(self.root.glob("*"))
        self.language_path = self.get_language(language)
        self.char_paths = list(self.language_path.glob("*"))

        self.set_size = set_size
        self.transform = transform

        self.enable_kshot = enable_kshot
        self.shot_size = shot_size

    def get_language(self, language: str | None = None) -> Path:
        languages = [lang.stem for lang in self.language_paths]

        if language is None or language == "":
            language = np.random.choice(self.language_paths)
        elif language not in languages:
            raise ValueError(
                f"{language} is not found, following languages are supported:",
                *languages,
            )
        else:
            language = self.root / language

        return language

    def pick_pair(self, idx: int) -> tuple[Path, Path]:
        """Pick random pair of images, if index is odd return same class
        otherwise different class

        Args:
            idx (int): sample index

        Returns:
            tuple[Path, Path]: character class paths
        """
        char1_path = np.random.choice(self.char_paths)

        if idx % 2 == 0:
            char2_path = np.random.choice(self.char_paths)
        else:
            char2_path = char1_path

        return char1_path, char2_path

    def __len__(self) -> int:
        return self.set_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enable_kshot:
            return self.get_kshot()
        else:
            return self.get_pair(idx)

    def get_pair(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        char1_path, char2_path = self.pick_pair(idx)

        label = torch.Tensor([int(char1_path == char2_path)])

        img1_path = np.random.choice(list(char1_path.glob("*.png")))
        img2_path = np.random.choice(list(char2_path.glob("*.png")))

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def get_kshot(self) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        query_char_path = np.random.choice(self.char_paths)

        query_img_path = np.random.choice(list(query_char_path.glob("*.png")))
        query_img = load_image(query_img_path)

        support_imgs = []
        for char_path in self.char_paths:
            support_img_paths = np.random.choice(
                list(char_path.glob("*.png")),
                self.shot_size,
            )
            support_img = [
                load_image(support_img_path) for support_img_path in support_img_paths
            ]
            support_imgs.append(support_img)

        if self.transform:
            query_img = self.transform(query_img)
            # num_class, shot_size
            support_imgs = torch.stack(
                [
                    torch.stack(
                        [
                            self.transform(this_class_support_img)
                            for this_class_support_img in support_img
                        ]
                    )
                    for support_img in support_imgs
                ]
            )

        return (
            query_img,
            support_imgs,
            self.char_paths.index(query_char_path),
        )
