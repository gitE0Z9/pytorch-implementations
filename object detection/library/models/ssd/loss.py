from functools import lru_cache

import torch
import torch.nn.functional as F
from constants.schema import DetectorContext
from models.ssd.anchor import PriorBox
from torch import nn
from torchvision.ops import box_convert, box_iou


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """logsoftmax"""
    # x is B*A,C+1
    x_max, _ = x.max(1, keepdim=True)
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def hard_negative_mining(
    conf_pred: torch.Tensor,
    matched_groundtruth: torch.Tensor,
    positives: torch.Tensor,
    num_negative: int,
):
    """for classification loss"""

    batch_size, num_boxes, num_classes = conf_pred.shape

    # Compute logsoftmax loss
    batch_conf = conf_pred.view(-1, num_classes)  # B*A, C+1
    cls_loss = log_sum_exp(batch_conf) - batch_conf.gather(
        1, matched_groundtruth[:, :, 4].view(-1, 1).long()
    )  # B*A, 1

    # Hard Negative Mining

    # positive box has highest log likelihood, least loss
    cls_loss = cls_loss.view(batch_size, num_boxes)  # B, A
    cls_loss = cls_loss * (1 - positives.long())

    _, loss_idx = cls_loss.sort(1, descending=True)  # B, A
    _, idx_rank = loss_idx.sort(1)  # B, A
    negatives = idx_rank < num_negative  # looking for higher quantile # B, A

    # Confidence Loss Including Positive and Negative Examples
    final_indicator = positives.logical_or(negatives)  # B, A

    cls_loss = F.cross_entropy(
        (conf_pred * final_indicator.unsqueeze(2)).view(-1, num_classes),
        (matched_groundtruth[:, :, 4] * final_indicator).view(-1).long(),
        reduction="sum",
    )

    return cls_loss


class MultiboxLoss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        negpos_ratio: float = 3,
        threshold: float = 0.5,
    ):
        super(MultiboxLoss, self).__init__()
        self.device = context.device
        self.negpos_ratio = negpos_ratio
        self.iou_threshold = threshold
        self.variances = [0.1, 0.2]

        self.anchors = PriorBox().anchors.to(context.device)

    @lru_cache
    def encode(self, gt: torch.Tensor, anchors: torch.Tensor):
        """encode gt loc information"""

        g_cxcy = (gt[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        g_wh = gt[:, 2:].log() - anchors[:, 2:].log()

        # encode variance, not in paper
        # g_cxcy /= self.variances[0] * anchors[:, 2:]
        # g_wh = torch.log(g_wh) / self.variances[1]

        return torch.cat([g_cxcy, g_wh], 1)

    @lru_cache
    def match(self, groundtruth: list) -> torch.Tensor:
        """matching between any truths and any priors"""
        # truths: ?,5, xyxy
        # placeholders: A,5, xywh
        # anchors: A,4, xywh
        batch_size = len(groundtruth)
        num_boxes = self.anchors.size(0)

        target = torch.zeros((batch_size, num_boxes, 5)).to(self.device)

        for batch_index, gt in enumerate(groundtruth):
            gt = torch.Tensor(gt).to(self.device)

            # ?, 8732
            overlaps = box_iou(
                box_convert(gt[:, :4], "xywh", "xyxy"),
                box_convert(self.anchors, "xywh", "xyxy"),
            )

            # ?, assign best anchor
            _, best_prior_idx = overlaps.max(1)

            # ?,4
            best_match_gt = self.encode(gt[:, :4], self.anchors[best_prior_idx])
            target[batch_index][best_prior_idx] = torch.cat(
                [best_match_gt, gt[:, 4:5]],
                1,
            )

            # A, assign gt to rest of anchor
            best_gt_overlap, best_gt_idx = overlaps.max(0)
            over_threshold = best_gt_overlap > self.iou_threshold

            # ??,4
            best_gt = self.encode(
                gt[:, :4][best_gt_idx[over_threshold]],
                self.anchors[over_threshold],
            )

            target[batch_index][over_threshold] = torch.cat(
                [best_gt, gt[best_gt_idx][over_threshold, 4:5]],
                1,
            )

        return target

    def forward(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        groundtruth: list,
    ):
        loc_pred, conf_pred = predictions

        target = self.match(groundtruth)

        positives = target[:, :, 4] > 0  # B, A
        num_positive = positives.sum(dim=1, keepdim=True)  # B, 1
        N = num_positive.sum()

        if N <= 0:
            return 0

        num_negative = torch.clamp(
            self.negpos_ratio * num_positive.long(),
            max=loc_pred.size(1) - 1,
        )  # B, 1

        # Localization Loss
        # B,num_pos,4
        loc_loss = F.smooth_l1_loss(
            loc_pred * positives.unsqueeze(2),
            target[:, :, :4] * positives.unsqueeze(2),
            reduction="sum",
        )

        cls_loss = hard_negative_mining(
            conf_pred,
            target,
            positives,
            num_negative,
        )

        # Sum of losses: L(c,l,g) = (Lconf(c, g) + αLloc(l, g)) / N
        loc_loss /= N
        cls_loss /= N
        return loc_loss + cls_loss
