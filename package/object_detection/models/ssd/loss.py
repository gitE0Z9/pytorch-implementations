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
    return (x - x_max).exp().sum(1, keepdim=True).log() + x_max


def hard_negative_mining(
    conf_pred: torch.Tensor,
    matched_gt: torch.Tensor,
    positive_mask: torch.Tensor,
    num_negative: int,
):
    """for classification loss"""

    batch_size, num_boxes, num_classes = conf_pred.shape

    # negative_mask = 1 - positive_mask
    # cls_loss = -(
    #     conf_pred[:, :, 1:].softmax(2).log().sum(2, keepdim=True) * positive_mask
    #     + conf_pred[:, :, 0:1].log() * negative_mask
    # )

    # Compute logsoftmax loss
    batch_conf = conf_pred.view(-1, num_classes)  # B*A, C+1
    cls_loss = log_sum_exp(batch_conf) - batch_conf.gather(
        1, matched_gt[:, :, 4].view(-1, 1).long()
    )  # B*A, 1

    # Hard Negative Mining

    # positive box has highest log likelihood, least loss
    cls_loss = cls_loss.view(batch_size, num_boxes, 1)  # B, A, 1
    cls_loss = cls_loss * (1 - positive_mask)

    _, loss_idx = cls_loss.sort(1, descending=True)  # B, A, 1
    _, idx_rank = loss_idx.sort(1)  # B, A, 1
    negative_mask = idx_rank < num_negative  # looking for higher quantile # B, A, 1

    # Confidence Loss Including Positive and Negative Examples
    final_indicator = positive_mask.logical_or(negative_mask)  # B, A, 1

    cls_loss = F.cross_entropy(
        (conf_pred * final_indicator).view(-1, num_classes),
        (matched_gt[:, :, 4:5] * final_indicator).view(-1).long(),
        reduction="sum",
    )

    return cls_loss


class MultiboxLoss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        negpos_ratio: float = 3,
        threshold: float = 0.5,
        alpha: float = 1,
    ):
        super(MultiboxLoss, self).__init__()
        self.device = context.device
        self.negpos_ratio = negpos_ratio
        self.iou_threshold = threshold
        self.alpha = alpha
        # self.variances = [0.1, 0.2]

        self.anchors = PriorBox().anchors.to(context.device)

    def encode(self, gt: torch.Tensor, anchors: torch.Tensor):
        """encode gt loc information"""

        g_cxcy = (gt[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        g_wh = gt[:, 2:].log() - anchors[:, 2:].log()

        # encode variance, not in paper
        # g_cxcy /= self.variances[0] * anchors[:, 2:]
        # g_wh = torch.log(g_wh) / self.variances[1]

        return torch.cat([g_cxcy, g_wh], 1)

    def match(self, groundtruth: list) -> torch.Tensor:
        """matching between any truths and any priors"""
        # truths: ?,5, xyxy
        # placeholders: A,5, xywh
        # anchors: A,4, xywh
        batch_size = len(groundtruth)
        num_boxes = self.anchors.size(0)

        converted_anchors = box_convert(self.anchors, "xywh", "xyxy")

        target = torch.zeros((batch_size, num_boxes, 5)).to(self.device)

        for batch_index, gt in enumerate(groundtruth):
            gt = torch.Tensor(gt).to(self.device)
            gt[:, 4] += 1

            # ?, 8732
            overlaps = box_iou(
                box_convert(gt[:, :4], "xywh", "xyxy"),
                converted_anchors,
            )

            # A, assign gt to rest of anchor
            best_gt_overlap, best_gt_idx = overlaps.max(0)
            over_threshold = best_gt_overlap > self.iou_threshold

            # ??,4
            gt[best_gt_idx[over_threshold], :4] = self.encode(
                gt[best_gt_idx[over_threshold], :4],
                self.anchors[over_threshold],
            )

            target[batch_index, over_threshold] = gt[best_gt_idx[over_threshold]]

            # ?, assign best anchor
            best_prior_idx = overlaps.argmax(1)

            # ?,4
            gt[:, :4] = self.encode(gt[:, :4], self.anchors[best_prior_idx])
            target[batch_index, best_prior_idx] = gt

        return target

    def forward(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        groundtruth: list,
    ):
        loc_pred, conf_pred = predictions

        _, num_boxes, _ = loc_pred.shape

        target = self.match(groundtruth)

        positive_mask = target[:, :, 4:5].gt(0).long()  # N, B, 1
        num_positive = positive_mask.sum(dim=1, keepdim=True)  # N, 1, 1

        N = positive_mask.sum()
        if N <= 0:
            return 0

        num_negative = torch.clamp(
            self.negpos_ratio * num_positive,
            max=num_boxes - 1,
        )  # N, 1, 1

        loc_loss = F.smooth_l1_loss(
            loc_pred * positive_mask,
            target[:, :, :4] * positive_mask,
            reduction="sum",
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
