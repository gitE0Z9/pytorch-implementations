from pathlib import Path

import numpy as np
import torch
from object_detection.constants.enums import OperationMode
from object_detection.datasets.coco.datasets import COCODatasetFromCSV
from object_detection.datasets.voc.datasets import VOCDatasetFromCSV


def dist_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        (1 - dist_metric(x[group == i], center))[:, i].mean().item()
        for i in range(cluster_num)
    ]
    iou = sum(within_cluster_iou) / cluster_num
    return iou


def kmeans(
    x: torch.Tensor,
    cluster_num: int = 5,
    total_iter: int = 300,
    error_acceptance: float = 1e-2,
) -> torch.Tensor:
    """generate kmeans index and center"""
    # init group
    init_point = torch.rand(cluster_num, 2)
    distance = dist_metric(x, init_point)
    group_index = distance.argmin(1)

    # iteratively update group
    prev_avg_iou = avg_iou(x, group_index, init_point)
    print("init mean IOU: ", prev_avg_iou)

    for _ in range(total_iter):
        for i in range(cluster_num):
            init_point[i] = x[group_index.eq(i)].mean(0)
        distance = dist_metric(x, init_point)
        group_index = distance.argmin(1)

        new_avg_iou = avg_iou(x, group_index, init_point)
        print("mean IOU: ", new_avg_iou)

        # early stopping
        if new_avg_iou - prev_avg_iou > error_acceptance:
            prev_avg_iou = new_avg_iou
        else:
            break

    # final group and cluster center
    return group_index, init_point


class PriorBox:
    def __init__(self, num_anchors: int, dataset: str):
        self.num_anchors = num_anchors
        self.dataset = dataset
        self.anchors_path = f"configs/yolov2/anchors.{dataset.lower()}.txt"

        if Path(self.anchors_path).exists():
            self.anchors = self.load_anchors()

    def load_anchors(self) -> torch.Tensor:
        anchors = np.loadtxt(self.anchors_path, delimiter=",")
        anchors = torch.from_numpy(anchors).float().view(1, len(anchors), 2, 1, 1)

        return anchors

    def build_anchors(self) -> torch.Tensor:
        dataset_class_mapping = {
            "VOC": VOCDatasetFromCSV,
            "COCO": COCODatasetFromCSV,
        }

        # comment lines of raw data
        dataset = dataset_class_mapping.get(self.dataset.upper())(
            mode=OperationMode.TRAIN.value,
            # transform=A.Compose(
            #     [
            #         A.Resize(100, 100),
            #         A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            #         ToTensorV2(),
            #     ]
            # ),
        )

        # dataloader = torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=32,
        #     collate_fn=collate_fn,
        #     shuffle=False,
        # )

        wh = torch.from_numpy(dataset.table[["w", "h"]].to_numpy())

        # wh = torch.cat(
        #     [
        #         torch.Tensor(label)[:, 2:4]
        #         for _, labels in tqdm(dataloader)
        #         for label in labels
        #     ],
        #     0,
        # )
        print("gt shape: ", wh.shape)

        # debug
        # wh = torch.rand(100,2)

        group_indices, anchors = kmeans(wh, self.num_anchors)

        final_avg_iou = (
            sum(
                [
                    (1 - dist_metric(wh[group_indices == i], anchors))[:, i]
                    .mean()
                    .item()
                    for i in range(self.num_anchors)
                ]
            )
            / self.num_anchors
        )
        print("final mean IOU: ", final_avg_iou)

        print(
            "member number in each group",
            (
                group_indices.view(-1, 1) == torch.arange(self.num_anchors).view(1, -1)
            ).sum(0),
        )

        return anchors

    def save_anchors(self, anchors: np.ndarray):
        with open(self.anchors_path, "w") as f:
            for w, h in anchors.tolist():
                print(f"{w},{h}", file=f)
