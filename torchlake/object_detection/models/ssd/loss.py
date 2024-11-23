import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert, box_iou

from ...constants.schema import DetectorContext
from ...utils.train import build_flatten_targets
from .anchor import load_anchors


def hard_negative_mining(
    conf_pred: torch.Tensor,
    matched_gt_class: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_ratio: float,
) -> torch.Tensor:
    """learn from top classification loss negative sample

    Args:
        conf_pred (torch.Tensor): confidence predictions, shape is (batch size, 8732, 1+c)
        matched_gt_class (torch.Tensor): class of groundtruth after matched and encoded, shape is (batch size, 8732)
        positive_mask (torch.Tensor): matched mask, shape is (batch size, 8732, 1)
        negative_ratio (float, optional): ratio of negative sample to positive sample. Defaults to 3.

    Returns:
        torch.Tensor: softmax loss
    """
    _, num_boxes, num_classes = conf_pred.shape
    # batch size, 1, 1
    num_positive = positive_mask.sum(dim=1, keepdim=True)

    # batch size, 1, 1
    num_negative = torch.clamp(
        negative_ratio * num_positive,
        # still left one positive sample
        max=num_boxes - 1,
    )

    # B, A, 1
    cls_loss = F.cross_entropy(
        conf_pred.permute(0, 2, 1),
        matched_gt_class.long(),
        reduction="none",
    ).unsqueeze(-1)

    # B*A [B*A] => 1
    positive_loss = cls_loss.view(-1)[
        positive_mask.view(-1).nonzero().squeeze_(-1)
    ].sum()

    # Hard Negative Mining

    with torch.no_grad():
        # negative box with topk confidence loss
        # B, A, 1
        all_negative_loss = cls_loss * (1 - positive_mask)

        # B, A, 1
        _, loss_idx = all_negative_loss.sort(1, descending=True)
        # B, A, 1
        _, loss_rank = loss_idx.sort()
        # B*A => ?
        negative_indices = (loss_rank < num_negative).view(-1).nonzero().squeeze_(-1)

    negative_loss = F.cross_entropy(
        # B, A, 1+c => B*A, 1+c => ?, 1+c
        conf_pred.view(-1, num_classes)[negative_indices],
        # ?
        # 0 is background class
        torch.zeros_like(negative_indices),
        reduction="sum",
    )

    return positive_loss + negative_loss


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        anchors: torch.Tensor = None,
        negative_ratio: float = 3,
        iou_threshold: float = 0.5,
        alpha: float = 1,
    ):
        """multibox loss

        Args:
            context (DetectorContext): detector context
            negative_ratio (float, optional): ratio of negative sample to positive sample. Defaults to 3.
            iou_threshold (float, optional): iou threshold to match. Defaults to 0.5.
            alpha (float, optional): classification loss coefficient. Defaults to 1.
        """
        super().__init__()
        self.device = context.device
        self.negative_ratio = negative_ratio
        self.iou_threshold = iou_threshold
        self.alpha = alpha
        # self.variances = [0.1, 0.2]

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

    def match(self, gt: list[torch.Tensor], anchors: torch.Tensor) -> torch.Tensor:
        """matching between any truths and any priors

        if anchor has overlap higher than threshold, then match
        if anchor has best overlap, then match

        Args:
            gt (list[torch.Tensor]): list of flatten target tensors, shape is batch size * (?, 5)
            anchors (torch.Tensor): anchors, shape is (8732, 4)

        Returns:
            torch.Tensor: matched targets, shape is (batch size, 8732, 5)
        """
        num_boxes = anchors.size(0)

        target = []
        for y in gt:
            # 8732, 5
            placeholder = torch.zeros(num_boxes, 5).to(self.device)

            # ?, 8732
            ious = box_iou(
                box_convert(y[:, :4], "cxcywh", "xyxy"),
                box_convert(anchors, "cxcywh", "xyxy"),
            )

            # shape is (8732,), assign gt to acceptable anchor
            best_gt_overlap, best_gt_idx = ious.max(0)
            over_threshold = best_gt_overlap > self.iou_threshold
            # shape is (??,)
            gt_idx_of_acceptable_prior = best_gt_idx[over_threshold]
            # ??,5
            placeholder[over_threshold] = torch.cat(
                [
                    self.encode(
                        y[gt_idx_of_acceptable_prior, :4],
                        anchors[over_threshold],
                    ),
                    y[gt_idx_of_acceptable_prior, 4:5],
                ],
                1,
            )

            # shape is (?,), assign best anchor
            best_prior_idx = ious.argmax(1)

            # ?,5
            placeholder[best_prior_idx] = torch.cat(
                [
                    self.encode(y[:, :4], anchors[best_prior_idx]),
                    y[:, 4:5],
                ],
                1,
            )

            target.append(placeholder)

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
        gt = gt.to(self.device).split(spans, 0)
        # shape is (batch size, 8732, 5)
        # unmatched will gain 0 as background
        target = self.match(gt, self.anchors)

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

        cls_loss = hard_negative_mining(
            conf_pred,
            target[:, :, 4],
            positive_mask,
            self.negative_ratio,
        )

        # Sum of losses: L = (L_conf + α L_loc) / N
        return (loc_loss + self.alpha * cls_loss) / N
