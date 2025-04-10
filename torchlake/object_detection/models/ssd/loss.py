import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert, box_iou

from ...constants.schema import DetectorContext
from ...utils.config import load_anchors
from ...utils.train import build_flatten_targets


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

    num_positive.squeeze_((-2, -1))
    remained_batch_indices = num_positive > 0

    # B, A, 1
    cls_loss = F.cross_entropy(
        conf_pred.permute(0, 2, 1),
        matched_gt_class.long(),
        reduction="none",
    ).unsqueeze(-1)

    # B*A [B*A] => ?
    positive_loss = cls_loss.view(-1)[positive_mask.view(-1).nonzero().squeeze_(-1)]

    positive_loss /= num_positive.repeat_interleave(num_positive)

    # Hard Negative Mining

    with torch.no_grad():
        # negative box with topk confidence loss
        # B, A, 1
        all_negative_loss = cls_loss * (1 - positive_mask)

        # B, A, 1
        _, loss_idx = all_negative_loss.sort(1, descending=True)
        # B, A, 1
        _, loss_rank = loss_idx.sort(1)
        # B*A => ?
        negative_indices = (loss_rank < num_negative).view(-1).nonzero().squeeze_(-1)

    negative_loss = F.cross_entropy(
        # B, A, 1+c => B*A, 1+c => ?, 1+c
        conf_pred.view(-1, num_classes)[negative_indices],
        # ?
        # 0 is background class
        torch.zeros_like(negative_indices),
        reduction="none",
    )

    negative_loss /= num_positive.repeat_interleave(num_negative.view(-1))

    return positive_loss.sum() + negative_loss.sum()


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

        # normalize by variance, not in paper
        # g_cxcy /= self.variances[0]
        # g_wh /= self.variances[1]

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
            over_threshold = best_gt_overlap > self.iou_threshold
            # shape is (?,)
            gt_idx_of_acceptable_prior = best_gt_idx[over_threshold]
            # ??,5
            placeholder[over_threshold] = torch.cat(
                [
                    self.encode(
                        gt[gt_idx_of_acceptable_prior, :4],
                        anchors[over_threshold],
                    ),
                    # 0 is background, so move class forward
                    gt[gt_idx_of_acceptable_prior, 4:5] + 1,
                ],
                1,
            )

            # shape is (num_gt,), assign best anchor
            best_prior_idx = iou.argmax(1)

            # num_gt,5
            placeholder[best_prior_idx] = torch.cat(
                [
                    self.encode(gt[:, :4], anchors[best_prior_idx]),
                    # 0 is background, so move class forward
                    gt[:, 4:5] + 1,
                ],
                1,
            )

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
        gt = gt.to(self.device)
        # shape is (batch size, 8732, 5)
        # unmatched will gain 0 as background
        target = self.match(gt, spans, self.anchors)

        # batch size, 8732
        positive_mask = target[:, :, 4].gt(0).long()

        # batch size
        N = positive_mask.sum(1)
        remained_batch_indices = N > 0

        # ? (all positive indices)
        positive_indices = positive_mask.view(-1).nonzero().squeeze_(-1)
        # ?, 4
        loc_loss = F.smooth_l1_loss(
            loc_pred[remained_batch_indices].view(-1, 4)[positive_indices],
            target[remained_batch_indices, :, :4].view(-1, 4)[positive_indices],
            reduction="none",
        )

        # 1
        loc_loss = torch.sum(loc_loss / N.repeat_interleave(N)[:, None])

        # 1
        cls_loss = hard_negative_mining(
            conf_pred,
            target[:, :, 4],
            positive_mask.unsqueeze(-1),
            self.negative_ratio,
        )

        # Sum of losses: L = (L_conf + α L_loc) / N
        return self.alpha * loc_loss + cls_loss
