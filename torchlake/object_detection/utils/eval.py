# import time
from typing import List, Tuple

import torch
import torchvision

# from torchlake.common.utils.image import load_image

from ..constants.enums import PRCurveInterpolation

# from ..utils.nms import greedy_nms


# def speed_evaluate(img_path: str, model_path: str):  # TODO: refactor
#     """evaluate inference, decode, nms time"""

#     testimg = load_image(img_path)

#     a = time.time()
#     testimgpred = model_predict(testimg)
#     b = time.time()
#     bbox_result = yolov2_decode(testimgpred, (IMAGE_SIZE, IMAGE_SIZE), NUM_CLASSES, 1)
#     c = time.time()
#     for class_idx in range(NUM_CLASSES):
#         best_index = greedy_nms(
#             bbox_result[0, :, :4],
#             bbox_result[0, :, 4] * bbox_result[0, :, 5 + class_idx],
#             NMS_THRESH,
#             CONF_THRESH,
#         )
#     d = time.time()

#     print("inference time: ", b - a)
#     print("decode time: ", c - b)
#     print("nms time: ", d - c)


def matched_gt_and_det(
    prediction: torch.Tensor,
    groundtruth: torch.Tensor,
    in_format: str = "xywh",
) -> torch.Tensor:
    """produce iou, prob, gt_num"""

    assert prediction.dim() == 2, "the pred should be [N,5]"
    assert groundtruth.dim() == 2, "the gt should be [M,4]"

    assert prediction.shape[1] == 5, "coord 4 and prob 1"
    assert groundtruth.shape[1] == 4, "coord 4"

    score = prediction[:, 4]  # ?

    if in_format == "xywh":
        prediction = torchvision.ops.box_convert(prediction[:, :4], "xywh", "xyxy")
        groundtruth = torchvision.ops.box_convert(groundtruth, "xywh", "xyxy")

    # if many prediction overlapped with a ground truth, then best prediction left
    # if many groundtruth overlapped with a prediction, not matter

    ious = torchvision.ops.box_iou(prediction, groundtruth)
    max_iou, gt_matched_index = ious.max(dim=1)
    iou_sorted, iou_sorted_index = max_iou.sort(descending=True)
    score = score[iou_sorted_index]
    gt_matched_index = gt_matched_index[iou_sorted_index]

    for i, matched_index in enumerate(gt_matched_index):
        iou_sorted[i + 1 :][gt_matched_index[i + 1 :] == matched_index] = 0

    return iou_sorted, score, groundtruth.size(0)  # float # float # int


def average_precision(
    table: List[tuple],
    iou_thres: float = 0.5,
    interpolation: str = PRCurveInterpolation.ALL.value,
) -> Tuple[List[float], List[float], float]:
    """calculate AP"""
    total_gt_num = sum(gt_num for _, _, gt_num in table)

    ious = [iou for iou, _, _ in table if iou is not None] or [torch.zeros(1, 1)]
    probs = [prob for _, prob, _ in table if prob is not None] or [torch.zeros(1, 1)]

    ious = torch.cat(ious, 0)
    probs = torch.cat(probs, 0)

    _, prob_index = probs.sort(descending=True)

    sorted_ious = ious[prob_index]  # positive samples

    tp = sorted_ious.gt(iou_thres).cumsum(0)  # true positive

    precision = tp / torch.arange(1, tp.size(0) + 1)
    recall = tp / total_gt_num

    mapping = {
        PRCurveInterpolation.ALL.value: 0,
        PRCurveInterpolation.VOC.value: 10,
        PRCurveInterpolation.COCO.value: 100,
    }

    precision, recall, auc = interpolate_pr_curve(
        precision,
        recall,
        mapping.get(interpolation),
        total_gt_num,
    )

    return precision, recall, auc


def interpolate_pr_curve(
    raw_precision: torch.Tensor,
    raw_recall: torch.Tensor,
    num_points: int,
    gt_num: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """convert raw pr curve to all interpolated one"""
    interpolated_precision = []

    recall = (
        torch.arange(0, 1 + 1 / num_points, 1 / num_points)
        if num_points > 0
        else raw_recall
    )

    for flag in recall:
        is_recall_in_range = raw_recall.ge(flag)
        precision_i_prime = (
            raw_precision[is_recall_in_range].max() if is_recall_in_range.sum() else 0
        )
        interpolated_precision.append(precision_i_prime)

    interpolated_precision = torch.Tensor(interpolated_precision)

    if num_points > 0:
        auc = interpolated_precision.mean()
    else:
        auc = interpolated_precision.sum() / gt_num

    return interpolated_precision, recall, auc.item()
