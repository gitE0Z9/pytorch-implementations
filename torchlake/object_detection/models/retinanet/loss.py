from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert, box_iou, sigmoid_focal_loss

from ...constants.schema import DetectorContext
from ...utils.config import load_anchors
from ...utils.train import build_flatten_targets


def focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: Literal["sum", "mean"] | None = "sum",
) -> torch.Tensor:
    # B, C, S # B, 1, S => B, 1，S
    pt = pred.gather(1, gt.unsqueeze(1))
    # B, 1，S
    l = -((1 - pt) ** gamma) * pt.log() * alpha

    if reduction == "sum":
        return l.sum()
    elif reduction == "mean":
        return l.mean()
    elif reduction == None:
        return l
    else:
        raise ValueError


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: Literal["sum", "mean"] | None = "sum",
    ):
        """_summary_

        Args:
            alpha (float, optional): weighting factor of classes. Defaults to 0.25.
            gamma (float, optional): penalty factor, in the interval of [0, 5]. Defaults to 2.
            reduction (Literal[&quot;sum&quot;, &quot;mean&quot;] | None, optional): _description_. Defaults to "sum".
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pytorch style
        return focal_loss(pred, gt, self.alpha, self.gamma, self.reduction)


class RetinaNetLoss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        anchors: torch.Tensor = None,
        iou_threshold: float = 0.5,
        ignore_iou_threshold: float = 0.4,
    ):
        super().__init__()
        self.device = context.device
        self.iou_threshold = iou_threshold
        self.ignore_iou_threshold = ignore_iou_threshold

        self.anchors = anchors
        if anchors is None:
            self.anchors = load_anchors(context.anchors_path).to(self.device)

    def encode(self, gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """encode gt loc information

        Args:
            gt (torch.Tensor): flatten target tensors, shape is (?, 5)
            anchors (torch.Tensor): anchors, shape is (8732, 4)

        Returns:
            torch.Tensor: target tensors
        """

        g_cxcy = (gt[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        g_wh = gt[:, 2:].log() - anchors[:, 2:].log()

        # assert not torch.isnan(g_cxcy).any()
        # assert not torch.isnan(g_wh).any()

        # encode variance, not in paper
        # g_cxcy /= self.variances[0] * anchors[:, 2:]
        # g_wh = torch.log(g_wh) / self.variances[1]

        return torch.cat([g_cxcy, g_wh], -1)

    def match(
        self,
        gt_batch: torch.Tensor,
        spans: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """matching between any truths and any priors

        if anchor has overlap higher than threshold, then match
        if anchor has best overlap, then match

        Args:
            gt_batch (torch.Tensor): flatten target tensors, shape is (?, 5)
            spans (torch.Tensor): number of gt of each image, shape is (batch size,)
            anchors (torch.Tensor): anchors, shape is (8732, 4)

        Returns:
            torch.Tensor: matched targets, shape is (batch size, 8732, 5)
        """
        num_boxes = anchors.size(0)

        # ?, 8732
        ious = box_iou(
            box_convert(gt_batch[:, :4], "cxcywh", "xyxy"),
            box_convert(anchors, "cxcywh", "xyxy"),
        )

        target = []
        offset = 0
        for span in spans:
            # num_gt, 8732
            iou = ious[offset : offset + span]
            # num_gt, 5
            gt = gt_batch[offset : offset + span]

            # 8732, 5
            placeholder = torch.zeros(num_boxes, 5).to(self.device)

            # shape is (8732,), assign gt to acceptable anchor
            best_gt_overlap, best_gt_idx = iou.max(0)
            over_threshold = best_gt_overlap >= self.iou_threshold
            # shape is (?,)
            gt_idx_of_acceptable_prior = best_gt_idx[over_threshold]
            # ??,5
            placeholder[over_threshold] = torch.cat(
                [
                    self.encode(
                        gt[gt_idx_of_acceptable_prior, :4],
                        anchors[over_threshold],
                    ),
                    gt[gt_idx_of_acceptable_prior, 4:5],
                ],
                1,
            )

            # ignore
            # ignore_threshold = best_gt_overlap > self.ignore_iou_threshold
            # ignore_threshold = ignore_threshold.logical_and(~over_threshold)

            # shape is (num_gt,), assign best anchor
            # best_prior_idx = iou.argmax(1)

            # # num_gt,5
            # placeholder[best_prior_idx] = torch.cat(
            #     [
            #         self.encode(gt[:, :4], anchors[best_prior_idx]),
            #         gt[:, 4:5],
            #     ],
            #     1,
            # )

            target.append(placeholder)

            offset += span

        return torch.stack(target, 0)

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        gt: list[list[list[int]]],
    ) -> torch.Tensor:
        """forward

        Args:
            pred (tuple[torch.Tensor, torch.Tensor]): location pred, confidence pred
            gt (list[list[list[int]]]): groundtruth list

        Returns:
            torch.Tensor: loss
        """
        # batch size, num boxes, 4 # batch size, num boxes, 1+class
        loc_pred, conf_pred = pred[:, :, :4], pred[:, :, 4:]

        # shape is (B, 5), format is (cx, cy, w, h, p)
        gt, spans = build_flatten_targets(gt, delta_coord=False)
        # 0 is background, so move class forward
        gt[:, -1] += 1
        gt = gt.to(self.device)
        # shape is (batch size, 8732, 5)
        # unmatched will gain 0 as background
        target = self.match(gt, spans, self.anchors)

        # batch size, 8732, 1
        positive_mask = target[:, :, 4:5].gt(0).long()

        N = positive_mask.sum()
        if N <= 0:
            return 0

        positive_indices = positive_mask.view(-1).nonzero().squeeze_(-1)
        loc_loss = F.smooth_l1_loss(
            loc_pred.view(-1, 4)[positive_indices],
            target[:, :, :4].view(-1, 4)[positive_indices],
            reduction="sum",
        )

        cls_loss = focal_loss(
            conf_pred.permute(0, 2, 1),
            target[:, :, 4].long(),
            reduction="sum",
        )

        # Sum of losses: L = (L_conf + α L_loc) / N
        return (loc_loss + cls_loss) / N
