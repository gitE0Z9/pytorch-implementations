from typing import Tuple
import torch
import numpy as np
from torch import nn
from torch import functional as F
from torchvision.ops import box_iou, box_convert


def log_sum_exp(x):
    """logsoftmax"""
    # x is B*A,C+1
    x_max, _ = x.max(1, keepdim=True)
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def hard_negative_mining(conf_pred, placeholder, pos, num_neg):
    """for classification loss"""

    batch_size, num_priors, num_classes = conf_pred.shape

    # Compute logsoftmax loss
    batch_conf = conf_pred.view(-1, num_classes)  # B*A, C+1
    cls_loss = log_sum_exp(batch_conf) - batch_conf.gather(
        1, placeholder[:, :, 4].view(-1, 1).long()
    )  # B*A, 1
    # Hard Negative Mining
    cls_loss = cls_loss.view(batch_size, num_priors)  # B, A
    cls_loss = cls_loss * (
        1 - pos.long()
    )  # positive box has highest log likelihood, least loss
    _, loss_idx = cls_loss.sort(1, descending=True)  # B, A
    _, idx_rank = loss_idx.sort(1)  # B, A
    neg = idx_rank < num_neg  # looking for higher quantile # B, A

    # Confidence Loss Including Positive and Negative Examples
    final_indicator = pos.logical_or(neg)  # B, A

    cls_loss = F.cross_entropy(
        (conf_pred * final_indicator.unsqueeze(2)).view(-1, num_classes),
        (placeholder[:, :, 4] * final_indicator).view(-1).long(),
        reduction="sum",
    )

    return cls_loss


class MultiboxLoss(nn.Module):
    def __init__(self, negpos_ratio: float, threshold: float):
        super(MultiboxLoss, self).__init__()
        self.negpos_ratio = negpos_ratio
        self.threshold = threshold
        self.variance = [0.1, 0.2]

    def encode(self, matched, priors, variances):
        """encode gt loc information"""

        g_cxcy = (matched[:, :2] - priors[:, :2]) / priors[:, 2:]
        g_wh = matched[:, 2:].log() - priors[:, 2:].log()

        # encode variance
        #     g_cxcy /= (variances[0] * priors[:, 2:])
        #     g_wh = torch.log(g_wh) / variances[1]

        return torch.cat([g_cxcy, g_wh], 1)

    def match(self, truths, placeholders, anchors, threshold, variances):
        """matching between any truths and any priors"""
        # truths: ?,5, xyxy
        # placeholders: A,5, xywh
        # anchors: A,4, xywh

        overlaps = box_iou(
            truths[:, :-1], box_convert(anchors, "xywh", "xyxy")
        )  # ?, 8732
        tmp_truths = box_convert(truths[:, :-1], "xyxy", "xywh")

        # ?, assign anchor to gt
        best_prior_overlap, best_prior_idx = overlaps.max(1)

        first_truths = self.encode(
            tmp_truths, anchors[best_prior_idx], variances
        )  # ?,4
        placeholders[best_prior_idx] = torch.cat([first_truths, truths[:, 4:5]], 1)

        # A, assign gt to rest of anchor
        best_truth_overlap, best_truth_idx = overlaps.max(0)
        over_t = best_truth_overlap > threshold

        second_truths = self.encode(
            tmp_truths[best_truth_idx][over_t], anchors[over_t], variances
        )  # ??,4

        placeholders[over_t] = torch.cat(
            [second_truths, truths[best_truth_idx][over_t, 4:5]], 1
        )

    def build_targets(
        self,
        groundtruth: list,
        target_shape: Tuple[int],
    ) -> torch.Tensor:
        grid_y, grid_x = target_shape[3], target_shape[4]
        target = np.zeros(target_shape)

        for b_idx, boxes in enumerate(groundtruth):
            for box in boxes:
                cx, cy, w, h, c, _ = box
                x, y = cx % (1 / grid_x), cy % (1 / grid_y)
                x_ind, y_ind = int(cx * grid_x), int(cy * grid_y)  # cell position
                target[b_idx, :, 4, y_ind, x_ind] = 1
                target[b_idx, :, 0:4, y_ind, x_ind] = [x, y, w, h]
                target[b_idx, :, 5 + int(c), y_ind, x_ind] = 1

        target = torch.from_numpy(target)
        return target

    def forward(self, predictions, targets, anchors):
        loc_pred, conf_pred = predictions
        batch_size, num_priors, _ = loc_pred.shape

        # matching
        placeholder = torch.zeros(batch_size, num_priors, 5).to(device)
        # targets shape is B X (?,5) in xyxyc
        for b in range(batch_size):
            self.match(
                targets[b], placeholder[b], anchors, self.threshold, self.variance
            )

        pos = placeholder[:, :, 4] > 0  # B, A
        num_pos = pos.sum(dim=1, keepdim=True)  # B, 1
        num_neg = torch.clamp(
            self.negpos_ratio * num_pos.long(), max=num_priors - 1
        )  # B, 1

        # Localization Loss
        # B,num_pos,4
        loc_loss = F.smooth_l1_loss(
            loc_pred * pos.unsqueeze(2),
            placeholder[:, :, :4] * pos.unsqueeze(2),
            reduction="sum",
        )

        cls_loss = hard_negative_mining(conf_pred, placeholder, pos, num_neg)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.sum()
        loc_loss /= N
        cls_loss /= N
        return loc_loss, cls_loss
