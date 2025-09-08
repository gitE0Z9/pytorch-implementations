from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from torchlake.common.models.kmeans import KMeans

from ...constants.schema import DetectorContext
from ...utils.config import load_anchors


def iou_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """compute iou to each centroid"""
    # compute pairwise iou
    iou = []
    for i in range(y.size(0)):
        intersection_wh = torch.min(x, y[i])  # N, 2
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]  # N
        union = x[:, 0] * x[:, 1] + y[i, 0] * y[i, 1] - intersection  # N
        iou.append(intersection / union)
    iou = torch.stack(iou, 1)

    # convert to distance
    distance = 1 - iou
    return distance


def avg_iou(x: torch.Tensor, group: torch.Tensor, center: torch.Tensor) -> float:
    cluster_num = center.size(0)
    within_cluster_iou = [
        (1 - iou_dist(x[group == i], center))[:, i].mean().item()
        for i in range(cluster_num)
    ]
    iou = sum(within_cluster_iou) / cluster_num
    return iou


class PriorBox:
    def __init__(self, context: DetectorContext):
        assert isinstance(
            context.num_anchors, int
        ), "number of anchors in context should be an integer"

        self.anchors_path = Path(context.anchors_path)
        self.num_anchors = context.num_anchors
        self.context = context

    def build_anchors(
        self,
        wh: Iterable[tuple[int | float, int | float]],
    ) -> torch.Tensor:
        na = self.num_anchors

        wh: torch.Tensor = torch.Tensor(wh)
        print("gt shape: ", wh.shape)

        m = KMeans(na, dist_metric=iou_dist, eval_metric=avg_iou)
        group_indices = m.fit(wh)
        anchors = m.centroids

        final_avg_iou = (
            sum(
                [
                    (1 - iou_dist(wh[group_indices == i], anchors))[:, i].mean().item()
                    for i in range(na)
                ]
            )
            / na
        )
        print("final mean IOU: ", final_avg_iou)

        print(
            "member count of each group: ",
            (group_indices.view(-1, 1) == torch.arange(na).view(1, -1)).sum(0).tolist(),
        )

        return anchors

    def load_anchors(self) -> torch.Tensor:
        if self.anchors_path.exists() and self.anchors_path.is_file():
            anchors = load_anchors(self.anchors_path)
            anchors = anchors.view(1, len(anchors), 2, 1, 1)
            return anchors.to(self.context.device)
        else:
            print(f"Can't find anchor file to path {self.anchors_path}")

    def save_anchors(self, anchors: np.ndarray):
        with open(self.anchors_path, "w") as f:
            for w, h in anchors.tolist():
                print(f"{w},{h}", file=f)
