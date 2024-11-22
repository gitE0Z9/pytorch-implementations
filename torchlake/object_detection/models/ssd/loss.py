import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert, box_iou

from ...constants.schema import DetectorContext
from ...utils.train import build_flatten_targets
from .anchor import load_anchors


def hard_negative_mining(
    conf_pred: torch.Tensor,
    matched_gt: torch.Tensor,
    positive_mask: torch.Tensor,
    num_negative: int,
):
    """learn from top classification loss negative sample

    Args:
        conf_pred (torch.Tensor): confidence predictions, shape is (batch size, 8732, 1+c)
        matched_gt (torch.Tensor): groundtruth after matched and encoded, shape is (batch size, 8732, 5)
        positive_mask (torch.Tensor): matched mask, shape is (batch size, 8732, 1)
        num_negative (int): number of negative samples, shape is (batch size, 1, 1)

    Returns:
        _type_: _description_
    """

    batch_size, num_boxes, num_classes = conf_pred.shape

    # negative_mask = 1 - positive_mask
    # cls_loss = -(
    #     conf_pred[:, :, 1:].softmax(2).log().sum(2, keepdim=True) * positive_mask
    #     + conf_pred[:, :, 0:1].log() * negative_mask
    # )

    # Compute logsoftmax loss, shape is (B*A, C+1)
    batch_conf = conf_pred.view(-1, num_classes)

    # negative log likelihood
    # shape is (B*A, 1)
    cls_loss = torch.logsumexp(batch_conf, -1, keepdim=True) - batch_conf.gather(
        1, matched_gt[:, :, 4].view(-1, 1).ne(0).long()
    )

    # Hard Negative Mining

    # positive box has highest log likelihood, least loss
    cls_loss = cls_loss.view(batch_size, num_boxes, 1)  # B, A, 1
    cls_loss = cls_loss * (1 - positive_mask)

    # B, A, 1
    _, loss_idx = cls_loss.sort(1, descending=True)
    # B, A, 1
    _, idx_rank = loss_idx.sort(1)
    # looking for higher quantile , shape is (B, A, 1)
    negative_mask = idx_rank < num_negative

    # Confidence Loss Including Positive and Negative Examples
    sampled_indices = positive_mask.logical_or(negative_mask)  # B, A, 1

    cls_loss = F.cross_entropy(
        # B*A, 1+c
        (sampled_indices * conf_pred).view(-1, num_classes),
        # B*A
        (sampled_indices * matched_gt[:, :, 4:5]).view(-1).long(),
        reduction="sum",
    )

    return cls_loss


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
        # batch_size = len(gt)
        num_boxes = anchors.size(0)

        # 8732, 4
        anchors = box_convert(anchors, "cxcywh", "xyxy")

        target = []
        for y in gt:
            # 8732, 5
            placeholder = torch.zeros(num_boxes, 5).to(self.device)

            # ?, 8732
            ious = box_iou(box_convert(y[:, :4], "cxcywh", "xyxy"), anchors)

            # shape is (8732,), assign gt to acceptable anchor
            best_gt_overlap, best_gt_idx = ious.max(0)
            over_threshold = best_gt_overlap > self.iou_threshold
            # shape is (??,)
            gt_idx_of_acceptable_prior = best_gt_idx[over_threshold]
            # ??,4
            y[gt_idx_of_acceptable_prior, :4] = self.encode(
                y[gt_idx_of_acceptable_prior, :4],
                anchors[over_threshold],
            )
            placeholder[over_threshold] = y[gt_idx_of_acceptable_prior]

            # shape is (?,), assign best anchor
            best_prior_idx = ious.argmax(1)

            # ?,4
            y[:, :4] = self.encode(y[:, :4], anchors[best_prior_idx])
            placeholder[best_prior_idx] = y

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
        loc_pred, conf_pred = pred

        _, num_boxes, _ = loc_pred.shape

        # shape is (B, 5), format is (cx, cy, w, h, p)
        gt, spans = build_flatten_targets(gt, delta_coord=False)
        gt = gt.to(self.device).split(spans, 0)
        # shape is (batch size, 8732, 5)
        target = self.match(gt, self.anchors)

        # batch size, 8732, 1
        positive_mask = target[:, :, 4:5].gt(0).long()
        # batch size, 1, 1
        num_positive = positive_mask.sum(dim=1, keepdim=True)

        N = positive_mask.sum()
        if N <= 0:
            return 0

        loc_loss = F.smooth_l1_loss(
            positive_mask * loc_pred,
            positive_mask * target[:, :, :4],
            reduction="sum",
        )

        # batch size, 1, 1
        num_negative = torch.clamp(
            self.negative_ratio * num_positive,
            # still left one positive sample
            max=num_boxes - 1,
        )

        cls_loss = hard_negative_mining(
            conf_pred,
            target,
            positive_mask,
            num_negative,
        )

        # cls_loss = F.cross_entropy(
        #     conf_pred.view(-1, conf_pred.size(2)),
        #     target[:, :, 4].view(-1).long(),
        #     reduction="sum",
        # )

        # Sum of losses: L = (L_conf + α L_loc) / N
        return (loc_loss + self.alpha * cls_loss) / N
