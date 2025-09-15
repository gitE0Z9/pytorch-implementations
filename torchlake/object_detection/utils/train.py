from typing import Literal

import torch
from torchvision.ops import box_iou

from torchlake.common.utils.numerical import generate_grid
from torchvision.ops._utils import _loss_inter_union


def collate_fn(batch) -> tuple[torch.Tensor, list[list[list[int]]]]:
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    return torch.stack(images, 0), labels


def generate_grid_train(
    grid_x: int,
    grid_y: int,
    is_center: bool = False,
) -> torch.Tensor:
    """recover x,y coord since we use x,y offset

    Args:
        grid_x (int): grid size in x direction
        grid_y (int): grid size in y direction
        is_center (bool, optional): move to the center of cells. Defaults to False.

    Returns:
        torch.Tensor: grids, in shape of (1, 1, 2, grid_y, grid_x)
    """
    x_offset, y_offset = generate_grid(
        grid_x,
        grid_y,
        is_center=is_center,
        normalized=True,
    )

    # 2, 7, 7 => 1, 1, 2, 7, 7
    return torch.stack([x_offset, y_offset])[None, None, ...]


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
    # coord_conversion
    converted_pred_box = xywh_to_xyxy(pred_box)  # B, 5, 4, 13, 13
    converted_gt_box = xywh_to_xyxy(gt_box)  # B, 1, 4, 13, 13

    # find intersection
    x1 = torch.max(
        converted_pred_box[:, :, 0:1, :, :], converted_gt_box[:, :, 0:1, :, :]
    )
    y1 = torch.max(
        converted_pred_box[:, :, 1:2, :, :], converted_gt_box[:, :, 1:2, :, :]
    )
    x2 = torch.min(
        converted_pred_box[:, :, 2:3, :, :], converted_gt_box[:, :, 2:3, :, :]
    )
    y2 = torch.min(
        converted_pred_box[:, :, 3:4, :, :], converted_gt_box[:, :, 3:4, :, :]
    )

    # N, 5, 1, 7, 7
    intersection = (x2 - x1).clamp(min=0, max=1) * (y2 - y1).clamp(min=0, max=1)
    pred_area = pred_box[:, :, 2:3, :, :] * pred_box[:, :, 3:4, :, :]  # N, 5, 1, 7, 7
    gt_area = gt_box[:, :, 2:3, :, :] * gt_box[:, :, 3:4, :, :]  # N, 1, 1, 7, 7
    total_area = pred_area + gt_area - intersection  # N, 5, 1, 7, 7
    total_area = total_area.clamp(min=0, max=1)

    assert intersection.ge(0).all(), "intersection should be no less than 0"
    assert total_area.ge(0).all(), "total area should be no less than 0"

    # compute iou
    intersection[intersection.gt(0)] /= total_area[intersection.gt(0)]

    return intersection


def wh_iou(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """IOU with wh tensor

    Args:
        x (torch.Tensor): width, height tensor. in shape of (M, 2)
        y (torch.Tensor): width, height tensor. in shape of (N, 2)

    Returns:
        torch.Tensor: IOU, in shape of (M, N)
    """
    x_xyxy = torch.cat(
        (
            -x[:, 0:2] / 2,
            x[:, 0:2] / 2,
        ),
        1,
    )
    y_xyxy = torch.cat(
        (
            -y[:, 0:2] / 2,
            y[:, 0:2] / 2,
        ),
        1,
    )

    return box_iou(x_xyxy, y_xyxy)


# copied from torchvision.ops.diou_loss.py
def iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> torch.Tensor:
    intsct, union = _loss_inter_union(boxes1, boxes2)
    iou = intsct / (union + eps)

    loss = 1 - iou
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def build_grid_targets(
    gt_batch: list[list[list[int]]],
    target_shape: tuple[int],
) -> torch.Tensor:
    """build grid targets

    Args:
        gt_batch (list[list[list[int]]]): batch of groundtruth, in format of (cx, cy, w, h, p)
        target_shape (tuple[int]): output shape, e.g. (batch, #num anchor, #num class + 5, #grid, #grid)

    Returns:
        torch.Tensor: targets, in format of (dx, dy, w, h, c, p)
    """
    grid_y, grid_x = target_shape[-2], target_shape[-1]
    targets = torch.zeros(target_shape)

    for batch_idx, boxes in enumerate(gt_batch):
        for box in boxes:
            # cx, cy here is absolute coord
            cx, cy, w, h, c = box[:5]
            # center position translation
            dx, dy = cx % (1 / grid_x), cy % (1 / grid_y)
            # cell position
            x_ind, y_ind = int(cx * grid_x), int(cy * grid_y)
            targets[batch_idx, :, 0, y_ind, x_ind] = dx
            targets[batch_idx, :, 1, y_ind, x_ind] = dy
            targets[batch_idx, :, 2, y_ind, x_ind] = w
            targets[batch_idx, :, 3, y_ind, x_ind] = h
            targets[batch_idx, :, 4, y_ind, x_ind] = 1
            targets[batch_idx, :, 5 + int(c), y_ind, x_ind] = 1

    return targets.float()


def build_flatten_targets(
    gt_batch: list[list[list[int]]],
    grid_shape: tuple[int] = None,
    delta_coord: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """build flatten targets

    Args:
        gt_batch (list[list[list[int]]]): batch of groundtruth, in format of (cx, cy, w, h, c)
        grid_shape (tuple[int], optional): grid shape e.g. (grid_y, grid_x). Defaults to None.
        delta_coord (bool, optional): represent center coord in translation delta and grid index. Defaults to False.

    Returns:
        tuple[torch.Tensor, list[int]]:
        element1: batched bboxes, in format(dx, dy, grid_x, grid_y, w, h, c) if delta coord is true
        or in format of (cx, cy, w, h, c).
        element2: number of detections of each image
    """
    if delta_coord:
        assert (
            grid_shape is not None
        ), "must provide grid shape to calculate coordinate deltas"
        grid_y, grid_x = grid_shape

    batch, spans = [], []
    for boxes in gt_batch:
        box_count = len(boxes)

        # ?, 5
        boxes = torch.Tensor(boxes)
        if delta_coord:
            cx, cy = boxes[:, 0:1], boxes[:, 1:2]
            dx, dy = cx % (1 / grid_x), cy % (1 / grid_y)

            # cell position
            x_ind, y_ind = (cx * grid_x).int(), (cy * grid_y).int()

            boxes = torch.cat([dx, dy, x_ind, y_ind, boxes[:, 2:5]], 1)

        batch.append(boxes)
        spans.append(box_count)

    # ?, 7 || batch size
    return torch.concat(batch), spans
