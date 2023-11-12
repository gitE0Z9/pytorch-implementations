from typing import Tuple

import numpy as np
import torch


def collate_fn(batch):
    image_list = []
    label_list = []
    for img, label in batch:
        image_list.append(img)
        label_list.append(label)

    return torch.stack(image_list, 0), label_list


def generate_grid_train(grid_x: int, grid_y: int, center: bool = False) -> torch.Tensor:
    """recover x,y coord since we use x,y offset"""
    x_offset, y_offset = torch.meshgrid(
        torch.arange(grid_x) / grid_x,
        torch.arange(grid_y) / grid_y,
        indexing="xy",
    )
    if center:
        y_offset, x_offset = (y_offset + 0.5 / grid_y), (x_offset + 0.5 / grid_x)

    grid = torch.stack([x_offset, y_offset], 0)  # 2, 7, 7
    grid = grid.view(1, 1, 2, grid_y, grid_x)  # 1, 1, 2, 7 ,7

    return grid


def xywh_to_xyxy(coord: torch.Tensor) -> torch.Tensor:
    """convert xywh to xyxy"""
    b, na, _, h, w = coord.shape
    device = "cuda:0" if coord.is_cuda else "cpu"
    grid_for_train = generate_grid_train(w, h).repeat(b, na, 1, 1, 1).to(device)
    xy = coord[:, :, 0:2, :, :] + grid_for_train  # B, na, 2, 7, 7
    wh = coord[:, :, 2:4, :, :]  # B, na, 2, 7, 7

    return torch.cat([xy - wh / 2, xy + wh / 2], dim=2)  # B, na, 4, 7, 7


def IOU(pred_box: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    """
    input: x_offset y_offset wh format
    output: iou for each batch and grid cell
    """

    # prevent zero division since most of grid cell are zero area
    epsilon = 1e-10

    # coord_conversion
    converted_pred_box = xywh_to_xyxy(pred_box)  # B, 5, 4, 13, 13
    converted_gt_box = xywh_to_xyxy(gt_box)  # B, 1, 4, 13, 13

    # find intersection
    x1 = torch.max(converted_pred_box[:, :, 0:1, :, :], converted_gt_box[:, :, 0:1, :, :])
    y1 = torch.max(converted_pred_box[:, :, 1:2, :, :], converted_gt_box[:, :, 1:2, :, :])
    x2 = torch.min(converted_pred_box[:, :, 2:3, :, :], converted_gt_box[:, :, 2:3, :, :])
    y2 = torch.min(converted_pred_box[:, :, 3:4, :, :], converted_gt_box[:, :, 3:4, :, :])

    intersection = (x2 - x1).clamp(min=0, max=1) * (y2 - y1).clamp(min=0, max=1)  # N, 5, 1, 7, 7
    pred_area = pred_box[:, :, 2:3, :, :] * pred_box[:, :, 3:4, :, :]  # N, 5, 1, 7, 7
    gt_area = gt_box[:, :, 2:3, :, :] * gt_box[:, :, 3:4, :, :]  # N, 1, 1, 7, 7
    total_area = pred_area + gt_area - intersection  # N, 5, 1, 7, 7
    total_area = total_area.clamp(min=0, max=1)

    assert intersection.ge(0).all(), "intersection should be no less than 0"
    assert total_area.ge(0).all(), "total area should be no less than 0"

    # compute iou
    intersection[intersection.gt(0)] /= total_area[intersection.gt(0)]

    return intersection


def build_targets(groundtruth_batch: list, target_shape: Tuple[int]) -> torch.Tensor:
    grid_y, grid_x = target_shape[-2], target_shape[-1]
    targets = np.zeros(target_shape)

    for batch_idx, boxes in enumerate(groundtruth_batch):
        for box in boxes:
            cx, cy, w, h, c = box[:5]
            x, y = cx % (1 / grid_x), cy % (1 / grid_y)
            x_ind, y_ind = int(cx * grid_x), int(cy * grid_y)  # cell position
            targets[batch_idx, :, 4, y_ind, x_ind] = 1
            targets[batch_idx, :, 0:4, y_ind, x_ind] = [x, y, w, h]
            targets[batch_idx, :, 5 + int(c), y_ind, x_ind] = 1

    return torch.from_numpy(targets)
