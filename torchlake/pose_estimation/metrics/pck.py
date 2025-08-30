import math
from typing import Literal, Sequence
import numpy as np
import torch


class PCK:

    def __init__(
        self,
        output_size: int,
        threshold_type: Literal["head", "torso", "image"] = "head",
        threshold: float = 0.5,
        visible_only: bool = False,
        image_size: int | None = None,
        head_top_index: int | None = None,
        neck_index: int | None = None,
        left_shoulder_index: int | None = None,
        right_hip_index: int | None = None,
    ):
        if threshold_type == "head":
            assert (
                head_top_index is not None and neck_index is not None
            ), "to use head as threshold, head top and neck must be provided"
            self.head_top_index = head_top_index
            self.neck_index = neck_index
        elif threshold_type == "torso":
            assert (
                left_shoulder_index is not None and right_hip_index is not None
            ), "to use torso as thresolf, left shoulder and right hip must be provided"
            self.left_shoulder_index = left_shoulder_index
            self.right_hip_index = right_hip_index
        elif threshold_type == "image":
            assert (
                image_size is not None
            ), "to use image as threshold, image size must be provided"
            self.image_size = image_size
        else:
            raise NotImplementedError

        self.output_size = output_size
        self.threshold_type = threshold_type
        self.threshold = threshold
        self.visible_only = visible_only

        # matched, total
        self.hits = torch.zeros(output_size, 2)

    def update(self, gt: torch.Tensor, pred: torch.Tensor):
        """update hits

        Args:
            gt (torch.Tensor): shape B, C, 2/3 (x, y, mask)
            pred (torch.Tensor): shape B, C, 2
        """
        gt = gt.float()

        # B, C
        dist = torch.norm(gt[:, :, :2] - pred.float(), p=2, dim=-1)

        if self.threshold_type == "image":
            threshold = self.threshold * self.image_size
        elif self.threshold_type == "head":
            # B, 1
            threshold = self.threshold * torch.norm(
                gt[:, self.head_top_index, :2] - gt[:, self.neck_index, :2],
                p=2,
                dim=-1,
                keepdim=True,
            )
        elif self.threshold_type == "torso":
            # B, 1
            threshold = self.threshold * torch.norm(
                gt[:, self.left_shoulder_index, :2] - gt[:, self.right_hip_index, :2],
                p=2,
                dim=-1,
                keepdim=True,
            )
        else:
            raise NotImplementedError

        hits = dist.lt(threshold)

        if self.visible_only:
            # mask invisible
            visible = 1 - gt[:, :, 2]
            self.hits[:, 0] += (hits * visible).sum(0)
            self.hits[:, 1] += visible.sum(0)
        else:
            self.hits[:, 0] += hits.sum(0)
            self.hits[:, 1] += hits.size(0)

    def get_total_accuracy(self) -> float:
        return (self.hits[:, 0] / self.hits[:, 1]).mean().item()

    def get_per_class_accuracy(self) -> np.ndarray:
        return (self.hits[:, 0] / self.hits[:, 1]).numpy()

    def show_per_class_accuracy(self, class_names: list[str]):
        for class_name, class_acc in zip(class_names, self.get_per_class_accuracy()):
            print(f"{class_name:<10}: {class_acc:.2f}")
