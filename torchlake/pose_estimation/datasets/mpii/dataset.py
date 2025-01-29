import json
from pathlib import Path
from typing import Literal

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchlake.common.utils.file import (
    read_json_file,
    read_matlab_file,
    write_json_file,
)
from torchlake.common.utils.image import load_image
from tqdm import tqdm

from .constants import MPII_CLASS_NAMES


class MPIIFromRaw(Dataset):
    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "test"],
        class_names: list[str] = MPII_CLASS_NAMES,
        crop_by_person: bool = True,
        transform=None,
    ):
        self.root = Path(root)
        self.mode = mode
        self.class_names = class_names
        self.crop_by_person = crop_by_person
        self.transform = transform

        file_handle = read_matlab_file(
            self.root.joinpath("mpii_human_pose_v1_u12_1.mat")
        )
        labels = file_handle["RELEASE"]["annolist"][0][0][0]
        training_idx = file_handle["RELEASE"]["img_train"][0][0][0]
        labels = (
            labels[training_idx == 1]
            if self.mode == "train"
            else labels[training_idx != 1]
        )

        # furthur clean labels
        if self.mode == "train":
            dropped = []
            # remove labels without annotations
            for i, l in enumerate(labels["annorect"]):
                try:
                    l["annopoints"]
                except:
                    dropped.append(i)

            # remove labels with empty points
            for i, l in enumerate(labels["annorect"]):
                if i not in dropped:
                    for k in l["annopoints"][0]:
                        try:
                            k["point"]
                        except:
                            dropped.append(i)
                            break

            print("dropped:", len(dropped))
            self.labels = np.delete(labels, dropped)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """get image and keypoint annotations from dataset
        1. raw image/annotations
        2. crop each person image/annotations
        3. transformed 1. or 2.

        Args:
            idx (_type_): _description_

        Raises:
            IndexError: _description_

        Returns:
            _type_: _description_
        """
        if idx >= len(self):
            raise IndexError(f"invalid index {idx}")

        image = self.get_image(idx)
        NUM_JOINTS = len(self.class_names)
        annotations, masks = self.get_label(idx)
        annotations = torch.Tensor(annotations).view(-1, NUM_JOINTS, 2).int()

        if self.crop_by_person:
            num_person = annotations.shape[0]
            h, w, _ = image.shape
            rect = annotations[masks.expand_as(annotations) == 0]

            images = []
            labels = []
            for p_idx in range(num_person):
                rect = annotations[p_idx][
                    masks[p_idx].expand_as(annotations[p_idx]) == 0
                ].view(-1, 2)
                x1, y1 = rect.min(0).values.tolist()
                x2, y2 = rect.max(0).values.tolist()
                x1, y1, x2, y2 = (
                    max(x1 - 30, 0),
                    max(y1 - 30, 0),
                    min(x2 + 30, w),
                    min(y2 + 30, h),
                )
                cropped = image[y1:y2, x1:x2]
                label = annotations[p_idx]
                label[:, 0] -= x1
                label[:, 1] -= y1

                if self.transform:
                    transformed = self.transform(
                        image=cropped,
                        keypoints=label.tolist(),
                    )
                    images.append(transformed["image"])
                    labels.append(transformed["keypoints"])
                else:
                    images.append(cropped)
                    labels.append(label)

            if self.transform:
                images = torch.stack(images)
                annotations = torch.Tensor(labels)
            else:
                annotations = torch.stack(labels)

            annotations = torch.cat([annotations, masks], -1)
            # raw: list[tensor], tensor(num person, num joint, dim)
            # transformed: tensor(num person, c, h, w), tensor
            return images, annotations

        # whole image and annotations
        # either transformed or not
        if self.transform:
            transformed = self.transform(
                image=image,
                keypoints=annotations.view(-1, 2),
            )
            image = transformed["image"]
            annotations = transformed["keypoints"]

        annotations = torch.Tensor(annotations).view(-1, NUM_JOINTS, 2)
        # num_person, num_joint, 3
        annotations = torch.cat([annotations, masks], -1)
        return image, annotations

    def get_image_path(self, idx: int) -> Path:
        image = self.labels["image"][idx]["name"][0][0][0]
        image = self.root.joinpath("data").joinpath(image)

        return image

    def get_image(self, idx: int) -> np.ndarray:
        image = self.get_image_path(idx)
        image = load_image(image, is_numpy=True)

        return image

    def get_label(self, idx: int) -> tuple[list[list[tuple[int, int]]], torch.Tensor]:
        # num_person, num_joint, 2
        NUM_JOINTS = len(self.class_names)
        keypoints_set = self.labels["annorect"][idx]["annopoints"][0]
        annotations = []
        masks = []
        head_top = self.class_names.index("head_top")
        for p_idx, keypoints in enumerate(keypoints_set):
            annotation = [(0, 0) for _ in range(NUM_JOINTS)]
            mask = [True for _ in range(NUM_JOINTS)]
            keypoints = keypoints["point"][0][0][0]

            try:
                keypoints["is_visible"]
            except:
                continue

            for x, y, c, is_visible in zip(
                keypoints["x"],
                keypoints["y"],
                keypoints["id"],
                keypoints["is_visible"],
            ):
                # https://stackoverflow.com/questions/59772847/mpii-pose-estimation-dataset-visibility-flag-not-present
                # occluded: empty
                # invisible: [["0"]]
                # visible: [["1"]]
                c_idx = c[0][0]
                occuluded = len(is_visible) == 0
                visible = not occuluded and (is_visible[0][0] in ["1", 1])
                annotation[c_idx] = (x[0][0], y[0][0])
                mask[c_idx] = (not visible) and (c_idx != head_top)
            annotations.append(annotation)
            masks.append(mask)

        # num_person, num_joint
        masks = torch.Tensor(masks).unsqueeze_(-1)

        return annotations, masks

    def to_json(self, path: Path | str):
        items = []
        for i in range(self.__len__()):
            ann, mask = self.get_label(i)
            items.append(
                {
                    "image": self.get_image_path(i).name,
                    "annotations": np.array(ann).astype(np.int32).tolist(),
                    "masks": np.array(mask).astype(np.int32).tolist(),
                }
            )

        write_json_file(path, items)

    def to_lmdb(self, env: lmdb.Environment):
        with env.begin(write=True) as tx:
            count = 0
            for imgs, labels in tqdm(self):
                labels = labels.tolist()
                if not self.crop_by_person:
                    tx.put(f"{count}".encode("utf-8"), imgs.tobytes())
                    tx.put(
                        f"{count}_shape".encode("utf-8"),
                        str(list(imgs.shape)).encode("utf-8"),
                    )
                    tx.put(
                        f"{count}_label".encode("utf-8"), str(labels).encode("utf-8")
                    )
                    count += 1
                else:
                    for p_idx, img in enumerate(imgs):
                        tx.put(f"{count}".encode("utf-8"), img.tobytes())
                        tx.put(
                            f"{count}_shape".encode("utf-8"),
                            str(list(img.shape)).encode("utf-8"),
                        )
                        tx.put(
                            f"{count}_label".encode("utf-8"),
                            str(labels[p_idx]).encode("utf-8"),
                        )
                        count += 1

            tx.put(b"count", str(count).encode("utf-8"))


class MPIIFromJSON(MPIIFromRaw):
    def __init__(
        self,
        root: str | Path,
        json_path: str | Path,
        class_names: list[str] = MPII_CLASS_NAMES,
        crop_by_person: bool = True,
        transform=None,
    ):
        self.root = Path(root)
        self.json_path = Path(json_path)
        self.class_names = class_names
        self.crop_by_person = crop_by_person
        self.transform = transform

        self.labels = read_json_file(json_path)

    def get_image_path(self, idx: int) -> Path:
        image = self.labels[idx]["image"]
        image = self.root.joinpath("data").joinpath(image)

        return image

    def get_label(
        self, idx: int
    ) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int]]]]:
        label = self.labels[idx]
        return torch.Tensor(label["annotations"]), torch.Tensor(label["masks"])


class MPIIFromLMDB(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        transform=None,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.transform = transform

        self.env = lmdb.open(lmdb_path)

        self.data_size = self.get_data_size()

    def get_data_size(self) -> int:
        with self.env.begin() as tx:
            return int(tx.get(b"count"))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if idx >= self.data_size:
            raise IndexError(f"invalid index {idx}")

        image = self.get_image(idx)
        label = self.get_label(idx)
        label = torch.Tensor(label)

        if self.transform:
            transformed = self.transform(
                image=image,
                keypoints=label[..., :2].tolist(),
            )
            image = transformed["image"]
            label[..., :2] = torch.Tensor(transformed["keypoints"])

        return image, label

    def get_image(self, idx: int) -> np.ndarray:
        with self.env.begin() as tx:
            shape = json.loads(tx.get(f"{idx}_shape".encode("utf-8")))
            img: np.ndarray = np.frombuffer(
                tx.get(f"{idx}".encode("utf-8")), np.uint8
            ).reshape(shape)

        return img

    def get_label(self, idx: int) -> list[list[list[int, int, int]]]:
        with self.env.begin() as tx:
            label = json.loads(tx.get(f"{idx}_label".encode("utf-8")))

        return label
