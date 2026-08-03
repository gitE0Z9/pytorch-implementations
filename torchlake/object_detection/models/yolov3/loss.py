from typing import Literal, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import (
    box_convert,
    box_iou,
    generalized_box_iou_loss,
    distance_box_iou_loss,
    complete_box_iou_loss,
)

from torchlake.object_detection.constants.schema import DetectorContext
from torchlake.object_detection.utils.train import (
    build_flatten_targets,
    generate_grid_train,
    wh_iou,
    iou_loss,
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
        loc_loss_type: Literal["mse", "diou", "ciou", "giou"] = "mse",
        cls_loss_type: Literal["softmax", "sigmoid"] = "sigmoid",
        label_smoothing: float = 0,
        return_all_loss: bool = False,
    ):
        super().__init__()
        assert anchors.device == torch.device(
            context.device
        ), "anchors should be on same device as criterion"
        # 1, A, 2(H, W), 1, 1
        self.anchors = anchors
        assert 0 <= label_smoothing <= 1, "label smoothing should fall in [0, 1]"

        self.num_anchors = context.num_anchors
        self.num_classes = context.num_classes
        self.device = context.device
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.iou_threshold = iou_threshold
        self.loc_loss_type = loc_loss_type
        self.cls_loss_type = cls_loss_type
        self.label_smoothing = label_smoothing
        self.return_all_loss = return_all_loss

    def encode(self, gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """encode gt loc information

        Args:
            gt (torch.Tensor): groundtruth coordinates, shape is (?, 4), format is (dx, dy, w, h)
            anchors (torch.Tensor): anchors, shape is (?, 2), format is (w, h)

        Returns:
            torch.Tensor: target tensors, shape is (?, 4)
        """

        g_dxdy = gt[:, :2]
        g_wh = gt[:, 2:4].log() - anchors.log()

        # NOTE: uncomment when loss is singular
        # assert not torch.isnan(g_dxdy).any()
        # assert not torch.isnan(g_wh).any()

        return torch.cat([g_dxdy, g_wh], -1)

    def match_anchor(self, gt_batch: list[list[list[float]]]) -> torch.Tensor:
        """match anchor to groundtruth

        Args:
            gt_batch (list[list[list[float]]]): batch of groundtruth, in shape of batch x number of bbox x (cx, cy, w, h, c)

        Returns:
            torch.Tensor: best prior indices for each groundtruth bbox, in shape of (?)
        """
        gt_wh: list[list[float]] = []
        for bboxes in gt_batch:
            for bbox in bboxes:
                gt_wh.append(bbox[2:4])

        # ?, A
        anchor_ious = wh_iou(
            torch.Tensor(gt_wh, device=self.device),
            self.anchors[0, :, :, 0, 0],
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
            gt_batch (torch.Tensor): a batch of groundtruth bboxes, in shape of (?, 7), in format of (dx, dy, grid_x, grid_y, w, h, c).
            spans (list[int]): number of annotated boxes in each image
            pred_batch (torch.Tensor): prediction, in shape of (B, A, (4+1+C), H, W)
            best_prior_indices (torch.Tensor): in shape of (?,)
            anchors(torch.Tensor): anchors, in format of (1, A, 2, 1, 1)
            grid_x (int): grid size along x dim
            grid_y (int): grid size along y dim
            anchor_indices (Sequence[int]): anchor indices of a head

        Returns:
            list[torch.Tensor]: a batch of encoded groundtruth objects. shape: B x (num_positive, C=7), C is (dx, dy, ln(w/a), ln(h/a), class, iou loss, best box index)
            torch.Tensor: matched or not. encoded as 0:negative 1:best 2:positive
        """
        is_iou_loss_on = self.loc_loss_type.endswith("iou")
        iou_loss_methods = {
            "iou": iou_loss,
            "diou": distance_box_iou_loss,
            "ciou": complete_box_iou_loss,
            "giou": generalized_box_iou_loss,
        }
        iou_loss_method = iou_loss_methods.get(self.loc_loss_type, None)

        # 1. convert gt from dxdywh into xyxy
        # gather dx, dy, w, h
        # in shape of (?, 4)
        # convert (dx, dy) to (cx, cy)
        xyxy_gt_batch = torch.cat(
            (
                gt_batch[:, 0:1] + gt_batch[:, 2:3] / grid_x,
                gt_batch[:, 1:2] + gt_batch[:, 3:4] / grid_y,
                gt_batch[:, 4:6],
            ),
            1,
        )
        xyxy_gt_batch = box_convert(xyxy_gt_batch, "cxcywh", "xyxy")

        # 2. convert prediction from dxdywh to xyxy
        with torch.set_grad_enabled(is_iou_loss_on):
            xyxy_preds = torch.cat(
                (
                    pred_batch[:, :, :2]
                    + generate_grid_train(grid_x, grid_y, is_center=False).to(
                        self.device
                    ),
                    self.anchors[:, anchor_indices] * pred_batch[:, :, 2:4].exp(),
                ),
                2,
            )
            # B*A*H*W, 4
            xyxy_preds = xyxy_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)
            xyxy_preds = box_convert(xyxy_preds, "cxcywh", "xyxy")

        # 3. match gt and pred image by image
        num_bboxes = grid_x * grid_y * len(anchor_indices)
        positive_targets = []
        positivities = []
        offset = 0
        for i, span in enumerate(spans):
            with torch.no_grad():
                # num_gt, A*H*W
                pred_iou = box_iou(
                    xyxy_gt_batch[offset : offset + span],
                    xyxy_preds[num_bboxes * i : num_bboxes * (i + 1)],
                )

                # A*H*W
                positivity = torch.zeros(num_bboxes).to(self.device)

                # shape is (A*H*W,), assign gt to acceptable anchor
                best_gt_overlap, _ = pred_iou.max(0)
                # shape is (A*H*W,)
                over_threshold = best_gt_overlap > self.iou_threshold
                # shape is (?,)
                positivity[over_threshold] = 2

                # shape: number of groundtruth object in an image, 7
                gt = gt_batch[offset : offset + span]
                best_prior_idx = best_prior_indices[offset : offset + span]
                mask = torch.logical_and(
                    min(anchor_indices) <= best_prior_idx,
                    best_prior_idx <= max(anchor_indices),
                )
                num_positive = mask.sum()
                if num_positive > 0:
                    best_box_idx = (
                        (best_prior_idx - min(anchor_indices)) * grid_x * grid_y
                        + gt[:, 3] * grid_x
                        + gt[:, 2]
                    ).long()[mask]
                    positivity[best_box_idx] = 1

            # shape: num_positive, 7
            # format: dx, dy, w, h, class, coef/iou_loss, best_box_idx
            positive_target = torch.zeros(num_positive, 7, device=self.device)
            if num_positive > 0:
                gt = gt[mask]
                positive_target[:, :4] = self.encode(
                    # shape: num_positive, 4
                    # format: dx, dy, w, h
                    gt[:, [0, 1, 4, 5]],
                    # shape: num_positive, 2
                    # format: w, h
                    self.anchors[0, best_prior_idx[mask], :, 0, 0],
                )
                positive_target[:, 4] = gt[:, 6]
                if is_iou_loss_on and iou_loss_method is not None:
                    with torch.set_grad_enabled(True):
                        positive_target[:, 5] = iou_loss_method(
                            xyxy_preds[best_box_idx + i * num_bboxes],
                            xyxy_gt_batch[offset : offset + span][mask],
                            reduction="sum",
                        )
                else:
                    # area normalizer = 2 - area
                    # used to normalize mse loss
                    positive_target[:, 5] = 2 - gt[:, 4:6].prod(1)
                # cast to float since placeholder is a float tensor
                positive_target[:, 6] = best_box_idx.float() + i * num_bboxes

            positive_targets.append(positive_target)
            positivities.append(positivity)

            offset += span

        # batch size x (num_positive, 7) # B, A*H*W
        return positive_targets, torch.stack(positivities)

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
            preds (list[torch.Tensor]): multi-scale predictions, in shape of (number of head) x (B, A*(4+1+C), H, W)
            gt (list[list[list[float]]]): batch of groundtruth, in shape of batch x number of bbox x (cx, cy, w, h, c)

        Returns:
            torch.Tensor: loss
        """
        positive_target_all, positivity_all, pred_all = [], [], []
        # ?
        best_prior_indices = self.match_anchor(gt)

        # loop over heads
        anchor_offset = 0
        num_boxes_offset = 0
        for pred, num_anchor in zip(preds, self.num_anchors):

            ### 1. decode batch of predicted bboxes in an image

            batch_size, channel, grid_y, grid_x = pred.shape
            pred: torch.Tensor = pred.unflatten(1, (num_anchor, channel // num_anchor))

            # transform
            pred[:, :, :2] = pred[:, :, :2].sigmoid()
            pred[:, :, 4] = pred[:, :, 4].sigmoid()

            if self.cls_loss_type == "softmax":
                pred[:, :, 5:] = pred[:, :, 5:].softmax(2)
            else:
                pred[:, :, 5:] = pred[:, :, 5:].sigmoid()

            ### 2. match groundtruth bboxes and predictions by IOU

            # shape is (?, 7), format is (dx, dy, grid_x, grid_y, w, h, c)
            flattened_gt, spans = build_flatten_targets(
                gt,
                (grid_y, grid_x),
                delta_coord=True,
            )
            flattened_gt = flattened_gt.to(self.device)

            # target shape is B x (num_positive, C=7)
            # C is dx, dy, w, h, class, area normalizer / iou loss, best box idx
            # positivity shape is B, A*H*W
            positive_target, positivity = self.match(
                flattened_gt,
                spans,
                pred[:, :, :4],
                best_prior_indices,
                grid_x,
                grid_y,
                tuple(range(anchor_offset, anchor_offset + num_anchor)),
            )
            # num_positives, 7
            positive_target = torch.cat(positive_target)
            positive_target[:, 6] += num_boxes_offset
            # B*A*H*W
            positivity = positivity.view(-1)

            # B*A*H*W, 5+C
            pred = (
                pred.permute(0, 1, 3, 4, 2).contiguous().view(-1, 5 + self.num_classes)
            )

            positive_target_all.append(positive_target)
            positivity_all.append(positivity)
            pred_all.append(pred)

            anchor_offset += num_anchor
            num_boxes_offset += batch_size * grid_y * grid_x * num_anchor

        # sum of (num_positive, 7)
        positive_target_all = torch.cat(positive_target_all)
        # sum of (B*A*H*W,)
        positivity_all = torch.cat(positivity_all)
        # sum of (B*A*H*W, 5+C)
        pred_all = torch.cat(pred_all)

        ### 3. compute losses

        # sum of (B*A*H*W,)
        negative_mask = positivity_all.eq(0)
        # no object loss for lower than threshold
        noobj_loss = F.mse_loss(
            pred_all[negative_mask, 4],
            torch.zeros(negative_mask.sum(), device=self.device),
            reduction="sum",
        )

        # good predictors
        # class loss / objecness loss / xywh loss
        best_indices = positive_target_all[:, 6].long()

        if self.loc_loss_type == "mse":
            coord_loss = (
                positive_target_all[:, 5:6]  # area normalizer
                * F.mse_loss(
                    pred_all[best_indices, :4],
                    positive_target_all[:, :4],
                    reduction="none",
                )
            ).sum()
        elif self.loc_loss_type.endswith("iou"):
            # iou loss
            coord_loss = positive_target_all[:, 5].sum()

        obj_loss = F.mse_loss(
            pred_all[best_indices, 4],
            torch.ones_like(best_indices).float(),
            reduction="sum",
        )

        cls_target = F.one_hot(
            positive_target_all[:, 4].long(), self.num_classes
        ).float()
        if self.label_smoothing > 0:
            cls_target[cls_target == 1] = 1 - self.label_smoothing
            cls_target[cls_target == 0] = self.label_smoothing / (self.num_classes - 1)
        cls_loss = F.mse_loss(
            pred_all[best_indices, 5:],
            cls_target,
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
