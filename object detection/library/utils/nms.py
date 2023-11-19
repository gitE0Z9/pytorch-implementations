from typing import List

import torch
import torchvision
from configs.schema import InferenceCfg
from numpy import intersect1d
from torchvision.ops.boxes import box_convert


def greedy_nms(
    bbox: torch.Tensor,
    score: torch.Tensor,
    nms_thres: float,
    conf_thres: float,
) -> List[int]:
    c_sort, sort_index = score.sort(descending=True)
    tmp_bbox = bbox[sort_index]
    tmp_bbox = torchvision.ops.box_convert(tmp_bbox, "xywh", "xyxy")

    best_index = []
    remove_index = []
    for i, (b, c) in enumerate(zip(tmp_bbox, c_sort)):
        if i == len(tmp_bbox) - 1:
            break
        if i in remove_index:
            continue
        if conf_thres >= c:
            continue
        best_index.append(sort_index[i].item())
        remove_index.append(i)
        other_b = tmp_bbox[(i + 1) :]
        iou = torchvision.ops.box_iou(b.unsqueeze(0), other_b)[0]
        over_index = iou.gt(nms_thres).nonzero().flatten().add(i + 1).tolist()
        remove_index.extend(over_index)

    return best_index


def soft_nms(
    bbox: torch.Tensor,
    score: torch.Tensor,
    nms_thres: float,
    conf_thres: float,
    sigma: float,
) -> List[int]:
    c_sort, sort_index = score.sort(descending=True)
    tmp_bbox = bbox[sort_index]
    tmp_bbox = torchvision.ops.box_convert(bbox, "xywh", "xyxy")
    N = tmp_bbox.size(0)
    tmp_score = score.clone()

    best_index = []
    remove_index = []
    for i in range(N):
        c, index = c_sort[i:].argmax(dim=0)
        b = tmp_bbox[index : (index + 1)]

        if i == N - 1:
            break
        if i in remove_index:
            continue
        if conf_thres >= c:
            continue
        best_index.append(sort_index[i].item())
        remove_index.append(i)
        mask = torch.Tensor([n not in remove_index for n in range(i + 1, N)]).int()
        other_b = tmp_bbox[(i + 1) :]
        iou = torchvision.ops.box_iou(b, other_b)[0]
        iou *= mask  # mask

        # gaussian kernel
        c_sort[(i + 1) :] *= torch.exp(-(iou**2) / sigma)
        tmp_score = c_sort

        # over_index = iou.gt(nms_thres).nonzero().flatten().add(i+1).tolist()
        # remove_index.extend(over_index)

    best_index = sort_index[tmp_score >= conf_thres]
    return best_index


def diou_nms(
    bbox: torch.Tensor,
    score: torch.Tensor,
    nms_thres: float,
    conf_thres: float,
    beta: float,
) -> List[int]:
    c_sort, sort_index = score.sort(descending=True)
    tmp_bbox = bbox[sort_index]
    tmp_bbox_xyxy = torchvision.ops.box_convert(tmp_bbox, "xywh", "xyxy")

    best_index = []
    remove_index = []
    for i, b in enumerate(tmp_bbox_xyxy):
        c = c_sort[i]
        if i == len(tmp_bbox_xyxy) - 1:
            break
        if i in remove_index:
            continue
        if conf_thres >= c:
            continue
        best_index.append(sort_index[i].item())
        remove_index.append(i)
        other_b = tmp_bbox_xyxy[(i + 1) :]
        box_c = torch.stack(
            [
                torch.minimum(b[0], other_b[:, 0]),
                torch.minimum(b[1], other_b[:, 1]),
                torch.maximum(b[2], other_b[:, 2]),
                torch.maximum(b[3], other_b[:, 3]),
            ],
            1,
        )
        iou = torchvision.ops.box_iou(b.unsqueeze(0), other_b)[0]
        d = (tmp_bbox[i, :2].unsqueeze(0) - tmp_bbox[(i + 1) :, :2]).pow(2).sum(1)
        c = (box_c[:, 0] - box_c[:, 2]).pow(2) + (box_c[:, 1] - box_c[:, 3]).pow(2)
        diou = iou - (d / c).pow(beta)

        over_index = diou.gt(nms_thres).nonzero().flatten().add(i + 1).tolist()
        remove_index.extend(over_index)

    return best_index


def confluence(
    bbox: torch.Tensor,
    score: torch.Tensor,
    confluence_thres: float,
    conf_thres: float,
    sigma: float,
) -> List[int]:
    c_sort, sort_index = score.sort(descending=True)
    tmp_bbox = bbox[sort_index]
    tmp_bbox = torchvision.ops.box_convert(tmp_bbox, "xywh", "xyxy")

    best_index = []
    remove_index = []
    for i, b in enumerate(tmp_bbox):
        c = c_sort[i]
        if i == len(tmp_bbox) - 1:
            break
        if i in remove_index:
            continue
        if conf_thres >= c:
            continue
        best_index.append(sort_index[i].item())
        remove_index.append(i)
        other_b = tmp_bbox[(i + 1) :]

        # normalize coordinates
        min_x, min_y, max_x, max_y = (
            torch.minimum(b[0], other_b[:, 0]),
            torch.minimum(b[1], other_b[:, 1]),
            torch.maximum(b[2], other_b[:, 2]),
            torch.maximum(b[3], other_b[:, 3]),
        )

        b = torch.stack(
            [
                (b[0] - min_x) / (max_x - min_x),
                (b[1] - min_y) / (max_y - min_y),
                (b[2] - min_x) / (max_x - min_x),
                (b[3] - min_y) / (max_y - min_y),
            ],
            1,
        )

        other_b[:, 0] = (other_b[:, 0] - min_x) / (max_x - min_x)
        other_b[:, 1] = (other_b[:, 1] - min_y) / (max_y - min_y)
        other_b[:, 2] = (other_b[:, 2] - min_x) / (max_x - min_x)
        other_b[:, 3] = (other_b[:, 3] - min_y) / (max_y - min_y)

        # compute proximity
        l1norm = (b - other_b).abs().sum(1)
        c_sort[(i + 1) :][l1norm.lt(confluence_thres)] = c_sort[(i + 1) :].mul(
            torch.exp(-(l1norm**2) / sigma)
        )[l1norm.lt(confluence_thres)]

        over_index = c_sort.lt(conf_thres).nonzero().flatten().add(i + 1).tolist()
        remove_index.extend(over_index)

    best_index = sort_index[c_sort >= conf_thres]

    return best_index


def fast_nms(
    bbox: torch.Tensor,
    score: torch.Tensor,
    nms_thres: float,
    conf_thres: float,
) -> torch.Tensor:
    c_sort, sort_index = score.sort(descending=True)
    tmp_bbox = torchvision.ops.box_convert(bbox, "xywh", "xyxy")[sort_index]

    iou = torchvision.ops.box_iou(tmp_bbox, tmp_bbox).triu_(diagonal=1)
    max_iou, max_iou_index = iou.max(dim=0)
    best_index = torch.logical_and(max_iou < nms_thres, c_sort >= conf_thres)
    best_index = sort_index[best_index]

    return best_index


def select_best_index(
    loc_detections: torch.Tensor,
    cls_detections: torch.Tensor,
    inference_config: InferenceCfg,
) -> List[int]:
    postprocess_method = inference_config.METHOD
    nms_thresh = inference_config.NMS_THRESH
    conf_thresh = inference_config.CONF_THRESH
    postprocess_parameter = inference_config.PARAMETER

    if postprocess_method == "greedy":
        best_index = greedy_nms(
            loc_detections,
            cls_detections,
            nms_thresh,
            conf_thresh,
        )
    elif postprocess_method == "soft":
        best_index = soft_nms(
            loc_detections,
            cls_detections,
            nms_thresh,
            conf_thresh,
            sigma=postprocess_parameter.SIGMA,
        )
    elif postprocess_method == "diou":
        best_index = diou_nms(
            loc_detections,
            cls_detections,
            nms_thresh,
            conf_thresh,
            beta=postprocess_parameter.BETA,
        )
    elif postprocess_method == "confluence":
        best_index = confluence(
            loc_detections,
            cls_detections,
            postprocess_parameter.CONFLUENCE_THRESH,
            conf_thresh,
            sigma=postprocess_parameter.SIGMA,
        )
    elif postprocess_method == "fast":
        best_index = fast_nms(
            loc_detections,
            cls_detections,
            nms_thresh,
            conf_thresh,
        )
    elif postprocess_method == "network":
        pass
    elif postprocess_method == "torchvision":
        converted_boxes = box_convert(loc_detections, "xywh", "xyxy")
        best_index = torchvision.ops.nms(converted_boxes, cls_detections, nms_thresh)
        cls_index = cls_detections.ge(conf_thresh).nonzero().view(-1).tolist()
        best_index = intersect1d(best_index, cls_index)
    # only confidence threshold
    elif postprocess_method == "":
        best_index = cls_detections.ge(conf_thresh).nonzero().view(-1).tolist()
    else:
        raise NotImplementedError

    return best_index
