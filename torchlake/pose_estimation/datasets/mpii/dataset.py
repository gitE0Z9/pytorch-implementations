from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchlake.common.utils.file import read_matlab_file
from torchlake.common.utils.image import load_image

from .constants import MPII_CLASS_NAMES


class MPIIFromRaw(Dataset):
    def __init__(
        self,
        root: str,
        mode: Literal["train", "test"],
        class_names: list[str] = MPII_CLASS_NAMES,
        single_person: bool = True,
        transform=None,
    ):
        self.root = Path(root)
        self.mode = mode
        self.class_names = class_names
        self.single_person = single_person
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"invalid index {index}")

        # image
        image = self.labels["image"][index]["name"][0][0][0]
        image = self.root.joinpath("data").joinpath(image)
        image = load_image(image, is_numpy=True)

        # num_person, num_joint, 2
        NUM_JOINTS = len(self.class_names)
        keypoints_set = self.labels["annorect"][index]["annopoints"][0]
        max_visible_points, max_idx = 0, 0
        annotations = []
        for p_idx, keypoints in enumerate(keypoints_set):
            annotation = [(0, 0) for _ in range(NUM_JOINTS)]
            visible_point = 0
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
                if len(is_visible) and (
                    is_visible[0][0] == 1 or is_visible[0][0] == "1"
                ):
                    annotation[c[0][0]] = (x[0][0], y[0][0])
                    visible_point += 1
            annotations.append(annotation)
            if visible_point > max_visible_points:
                max_visible_points = visible_point
                max_idx = p_idx

        # single person and multi person
        # and then transform

        if self.single_person:
            # for single person, choose most visible one
            # num_joint, 2
            annotations = annotations[max_idx]

        if self.transform:
            if not self.single_person:
                annotations = [ele for ele in annotation for annotation in annotations]
            transformed = self.transform(image=image, keypoints=annotations)

        annotations = torch.Tensor(transformed["keypoints"])
        if not self.single_person:
            annotations = annotations.view(-1, NUM_JOINTS, 2)

        return transformed["image"], annotations
