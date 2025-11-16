from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from torchlake.common.utils.image import load_image
from .constant import LANGUAGES


class OmniglotSet(Dataset):
    def __init__(
        self,
        root: str | Path,
        languages: str | Sequence[str] | None = None,
        transform: Callable | None = None,
        enable_pair: bool = False,
        enable_episode: bool = False,
        way_size: int = 1,
        shot_size: int = 5,
        query_size: int = 1,
        episode_size: int = 0,
    ):
        """OmniglotSet dataset

        Args:
            root (str | Path): the path to the root directory of OmniglotSet
            languages (str | Sequence[str] | None, optional): the path of directories of languages. Defaults to None.
            transform (Callable | None, optional): torchvision transform. Defaults to None.
            enable_pair (bool, optional): enable image pair of same class or different classes. Defaults to False.
            enable_episode (bool, optional): enable n-way, k-shot learning. Defaults to False.
            way_size (int, optional): n-way size. Defaults to 5.
            shot_size (int, optional): k-shot size. Defaults to 5.
            query_size (int, optional): query size. Defaults to 5.
            episode_size (int, optional): episode size for few shot learning. Defaults to 0.
        """
        self.root = Path(root) / "omniglot-py" / "images_background"
        self.set_languages(languages)

        self.transform = transform

        self.enable_pair = enable_pair
        self.enable_episode = enable_episode
        self.way_size = way_size
        self.shot_size = shot_size
        self.query_size = query_size
        self.episode_size = episode_size

        if self.enable_episode:
            assert self.way_size <= len(
                self.char_paths
            ), "way size should be less than or equal to character sizes"

    # @property
    # def chars(self) -> tuple[str]:
    #     return tuple(it.stem for it in self.char_paths)

    # self.languages, self.lang_chars, self.char_paths
    def set_languages(self, languages: str | Sequence[str] | None = None):
        """set languages

        Args:
            languages (str | Sequence[str] | None, optional): the path of directories of languages. Defaults to None.
        """
        if languages is None:
            self.languages = LANGUAGES
        elif isinstance(languages, str):
            self.languages = tuple(languages)
        else:
            self.languages = languages

        # lang: char paths
        lang_chars = {}
        char_paths = []
        for lang in self.languages:
            char_path = list(self.root.joinpath(lang).glob("*"))
            lang_chars[lang] = list(it.stem for it in char_path)
            char_paths.extend(char_path)

        self.char_paths: list[Path] = char_paths
        self.lang_chars: dict[str, tuple[str]] = lang_chars

    def __len__(self) -> int:
        return self.episode_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise StopIteration

        if self.enable_episode:
            return self.get_episode()
        elif self.enable_pair:
            return self.get_pair(idx)
        else:
            raise NotImplementedError(
                "image classification task should use torchvision.datasets.ImageFolder"
            )

    def get_pair(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # even => same, odd => different
        same_class = idx % 2 == 0
        char1_path, char2_path = self.pick_pair(same_class)

        label = torch.Tensor([int(same_class)])

        img1_path = np.random.choice(tuple(char1_path.glob("*.png")))
        img2_path = np.random.choice(tuple(char2_path.glob("*.png")))

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def pick_pair(self, same_class: bool = False) -> tuple[Path, Path]:
        """Pick random pair of classes

        Args:
            same_class (bool): two images are from the same class

        Returns:
            tuple[Path, Path]: paths to character classes
        """
        char1_path = np.random.choice(self.char_paths)

        if same_class:
            return char1_path, char1_path
        else:
            return char1_path, np.random.choice(self.char_paths)

    def get_episode(self) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        seleted_chars: list[Path] = np.random.choice(self.char_paths, self.way_size)

        # n x (#query x (h, w, c))
        query_sets = []
        # n x (k x (h, w, c))
        support_sets = []
        for selected_char in seleted_chars:
            query_sets.append([])
            support_sets.append([])
            selected_imgs_path = np.random.choice(
                tuple(selected_char.glob("*.png")),
                self.shot_size + self.query_size,
            )
            for selected_img_path in selected_imgs_path[: self.shot_size]:
                img = load_image(selected_img_path)
                support_sets[-1].append(img)
            for selected_img_path in selected_imgs_path[self.shot_size :]:
                img = load_image(selected_img_path)
                query_sets[-1].append(img)

        if self.transform:
            # n, #query, h, w, c
            query_sets = torch.stack(
                tuple(
                    torch.stack(tuple(self.transform(q) for q in query_set))
                    for query_set in query_sets
                )
            )
            # n, k, h, w, c
            support_sets = torch.stack(
                tuple(
                    torch.stack(tuple(self.transform(s) for s in support_set))
                    for support_set in support_sets
                )
            )

        return query_sets, support_sets
