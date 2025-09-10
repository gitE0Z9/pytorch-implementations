from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou

from torchlake.object_detection.constants.schema import DetectorContext
from torchlake.object_detection.utils.train import (
    build_flatten_targets,
    generate_grid_train,
    wh_iou,
)


class YOLOV3Loss(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        context: DetectorContext,
        lambda_obj: float = 1,
        lambda_noobj: float = 1,
        lambda_coord: float = 0.75,
        iou_threshold: float = 0.7,
        return_all_loss: bool = False,
    ):
        super().__init__()
        assert anchors.device == torch.device(
            context.device
        ), "anchors should be on same device as criterion"
        self.anchors = anchors

        self.num_anchors = context.num_anchors
        self.num_classes = context.num_classes
        self.device = context.device
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.iou_threshold = iou_threshold
        self.return_all_loss = return_all_loss

    def encode(self, gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """encode gt loc information

        Args:
            gt (torch.Tensor): groundtruth coordinates in format of (dx, dy, w, h), shape is (?, 4)
            anchors (torch.Tensor): anchors, shape is (num_boxes, 2)

        Returns:
            torch.Tensor: target tensors, shape is (num_boxes, 4)
        """

        g_dxdy = gt[:, :2]
        g_wh = gt[:, 2:].log() - anchors.log()

        # NOTE: uncomment when loss is singular
        # assert not torch.isnan(g_dxdy).any()
        # assert not torch.isnan(g_wh).any()

        return torch.cat([g_dxdy, g_wh], -1)

    def match_anchor(self, gt_batch: list[list[list[float]]]) -> torch.Tensor:
        output = []
        for b in gt_batch:
            for o in b:
                output.append(o[2:4])

        # ?, A
        anchor_ious = wh_iou(
            torch.Tensor(output).to(self.device),
            self.anchors[0, 2:4, :, 0, 0],
        )

        # shape is ?, assign best anchor to each gt
        return anchor_ious.argmax(1)

    def match(
        self,
        gt_batch: torch.Tensor,
        spans: list[int],
        pred_batch: torch.Tensor,
        best_prior_indices: torch.Tensor,
        grid_x: int,
        grid_y: int,
        anchor_indices: Sequence[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """match anchor to groundtruth
        1. if pred iou over threshold, don't update loss
        2. if not best anchor match and pred iou lower than threshold, update noobj loss
        3. update coord/obj/class loss for best anchor match

        Args:
            gt_batch (torch.Tensor): a batch of groundtruth bboxes, in shape of (?, 7), in format(dx, dy, grid_x, grid_y, w, h, c).
            spans (list[int]): number of annotated boxes in each image
            pred_batch (torch.Tensor): prediction, in format of (B, A, (4+1+C), H, W)
            anchors(torch.Tensor): anchors, in format of (1, A, 2, 1, 1)
            grid_x (int): grid size along x dim
            grid_y (int): grid size along y dim
            anchor_indices (Sequence[int]): anchor indices

        Returns:
            torch.Tensor: in shape of (B, A, H, W, C=7), C is dx, dy, ln(w/a), ln(h/a), class, iou, positive_level
            positive_level is encoded as 0:negative 1:best 2:positive
        """
        # 1. convert gt into cxcywh

        # gather dx, dy, w, h
        # in shape of (?, 4)
        # convert (dx, dy) to (cx, cy)

        # 2. convert prediction to xyxy
        gts = torch.cat(
            (
                gt_batch[:, 0:1] + gt_batch[:, 2:3] / grid_x,
                gt_batch[:, 1:2] + gt_batch[:, 3:4] / grid_y,
                gt_batch[:, 4:6],
            ),
            1,
        )
        gts = box_convert(gts, "cxcywh", "xyxy")

        preds = torch.cat(
            (
                pred_batch[:, :, :2]
                + generate_grid_train(grid_x, grid_y, is_center=False).to(self.device),
                self.anchors[:, anchor_indices] * pred_batch[:, :, 2:4].exp(),
            ),
            2,
        )
        # B*A*H*W, 4
        preds = preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)
        preds = box_convert(preds, "cxcywh", "xyxy")

        num_boxes = grid_x * grid_y * len(anchor_indices)
        target = []
        positivities = []
        # match gt and pred image by image
        offset = 0
        for i, span in enumerate(spans):
            # num_gt, A*H*W
            pred_iou = box_iou(
                gts[offset : offset + span],
                preds[num_boxes * i : num_boxes * (i + 1)],
            )

            # A*H*W
            positivity = torch.zeros(num_boxes).to(self.device)

            # shape is (A*H*W,), assign gt to acceptable anchor
            best_gt_overlap, _ = pred_iou.max(0)
            # shape is (A*H*W,), only ? is true
            over_threshold = best_gt_overlap > self.iou_threshold
            # shape is (?,)
            positivity[over_threshold] = 2

            # num_gt, 7
            gt = gt_batch[offset : offset + span]
            best_prior_idx = best_prior_indices[offset : offset + span]
            mask = torch.logical_and(
                min(anchor_indices) <= best_prior_idx,
                best_prior_idx <= max(anchor_indices),
            )
            best_box_idx = (
                best_prior_idx * grid_x * grid_y + gt[:, 3] * grid_x + gt[:, 2]
            ).long()[mask]
            positivity[best_box_idx] = 1

            # ?, 7
            # C is dx, dy, w, h, class, coef, best_box_idx
            placeholder = torch.zeros(mask.sum(), 7).to(self.device)
            if mask.sum() > 0:
                gt = gt[mask]
                placeholder[:, :4] = self.encode(
                    gt[:, [0, 1, 4, 5]],
                    self.anchors[0, best_prior_idx, :, 0, 0],
                )
                placeholder[:, 4] = gt[:, 6]
                placeholder[:, 5] = 2 - gt[:, 4:6].prod(1)
                placeholder[:, 6] = best_box_idx.float() + i * num_boxes

            target.append(placeholder)
            positivities.append(positivity)

            offset += span

        # ?, 7 # B, A*H*W
        return target, torch.stack(positivities)

    def forward(
        self,
        preds: list[torch.Tensor],
        gt: list[list[list[int]]],
    ) -> torch.Tensor:
        """forward function of YOLOv2Loss
        Some extra rules
        positive anchors: x,y,w,h,c,p loss
        before 12800, negative anchors : use anchors as truths
        best matched, iou lower than threshold: noobject loss
        best matched, iou over threshold: no loss

        p.s. match with fixed anchor and no overlapping groundtruth(?

        Args:
            preds (list[torch.Tensor]): multi-scale predictions, in format of (B, A*(4+1+C), H, W)
            gt (list[list[list[int]]]): batch of groundtruth, in format of (cx, cy, w, h, c)

        Returns:
            torch.Tensor: loss
        """
        target_all, positivity_all, pred_all = [], [], []
        best_prior_indices = self.match_anchor(gt)

        anchor_offset = 0
        num_boxes_offset = 0
        for pred, num_anchor in zip(preds, self.num_anchors):
            batch_size, channel, grid_y, grid_x = pred.shape
            pred: torch.Tensor = pred.unflatten(1, (num_anchor, channel // num_anchor))

            # transform
            pred[:, :, :2] = pred[:, :, :2].sigmoid()
            pred[:, :, 4:] = pred[:, :, 4:].sigmoid()

            with torch.no_grad():
                # shape is (B, 7), format is (dx, dy, grid_x, grid_y, w, h, c)
                flattened_gt, spans = build_flatten_targets(
                    gt, (grid_y, grid_x), delta_coord=True
                )
                flattened_gt = flattened_gt.to(self.device)

                # target shape is (?, C=7)
                # C is dx, dy, w, h, class, coef, best_box_idx
                # positivity shape is B, A*H*W
                target, positivity = self.match(
                    flattened_gt,
                    spans,
                    pred[:, :, :4],
                    best_prior_indices,
                    grid_x,
                    grid_y,
                    tuple(range(anchor_offset, anchor_offset + num_anchor)),
                )
                # ?, 7
                target = torch.cat(target)
                target[:, 6] += num_boxes_offset
                # B*A*H*W
                positivity = positivity.view(-1)

            # B*A*H*W, 5+C
            pred = (
                pred.permute(0, 1, 3, 4, 2).contiguous().view(-1, 5 + self.num_classes)
            )

            target_all.append(target)
            positivity_all.append(positivity)
            pred_all.append(pred)

            anchor_offset += num_anchor
            num_boxes_offset += batch_size * grid_y * grid_x * num_anchor

        target_all = torch.cat(target_all)
        positivity_all = torch.cat(positivity_all)
        pred_all = torch.cat(pred_all)

        # B*A*H*W
        negative_mask = positivity_all.eq(0)
        # no object loss for lower than threshold
        noobj_loss = F.mse_loss(
            pred_all[negative_mask, 4],
            torch.zeros(negative_mask.sum()).to(self.device),
            reduction="sum",
        )

        # good predictors
        # class loss / objecness loss / xywh loss
        # positive_mask = positivity.eq(2)
        # best_mask = positivity.eq(1)
        mask = target_all[:, 6].ne(0)
        target_all = target_all[mask]
        best_indices = target_all[:, 6].long()
        coord_loss = (
            target_all[:, 5:6]  # area normalizer
            * F.mse_loss(
                pred_all[best_indices, :4],
                target_all[:, :4],
                reduction="none",
            )
        ).sum()
        obj_loss = F.mse_loss(
            pred_all[best_indices, 4],
            torch.ones_like(best_indices).float(),
            reduction="sum",
        )
        cls_loss = F.mse_loss(
            pred_all[best_indices, 5:],
            F.one_hot(target_all[:, 4].long(), self.num_classes).float(),
            reduction="sum",
        )

        total_loss = (
            cls_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_obj * obj_loss
            + self.lambda_coord * coord_loss
        )

        if self.return_all_loss:
            return total_loss, cls_loss, noobj_loss, obj_loss, coord_loss
        else:
            return total_loss
