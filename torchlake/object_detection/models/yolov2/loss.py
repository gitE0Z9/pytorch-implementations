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


class YOLOV2Loss(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        context: DetectorContext,
        lambda_obj: float = 5,
        lambda_noobj: float = 1,
        lambda_prior: float = 0.01,
        lambda_coord: float = 1,
        iou_threshold: float = 0.6,
        prior_loss_threshold: int = 12800,
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
        self.lambda_prior = lambda_prior
        self.lambda_coord = lambda_coord
        self.iou_threshold = iou_threshold
        self.prior_loss_threshold = prior_loss_threshold
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

    def match(
        self,
        gt_batch: torch.Tensor,
        spans: list[int],
        pred_batch: torch.Tensor,
        grid_x: int,
        grid_y: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """match anchor to groundtruth
        1. if pred iou over threshold, don't update loss
        2. if not best anchor match and pred iou lower than threshold, update noobj loss
        2. before 12800 pictures, update prior loss
        3. update coord/obj/class loss for best anchor match

        Args:
            gt_batch (torch.Tensor): a batch of groundtruth bboxes, in shape of (?, 7), in format(dx, dy, grid_x, grid_y, w, h, c).
            spans (list[int]): number of annotated boxes in each image
            pred_batch (torch.Tensor): prediction, in format of (B, A, (4+1+C), H, W)
            grid_x (int): grid size along x dim
            grid_y (int): grid size along y dim

        Returns:
            torch.Tensor: in shape of (B, A, H, W, C=7), C is dx, dy, ln(w/a), ln(h/a), class, iou, positive_level
            positive_level is encoded as 0:negative 1:best 2:positive
        """
        # 1. get anchor iou

        # ?, A
        anchor_ious = wh_iou(gt_batch[:, 4:6], self.anchors[0, :, :, 0, 0])
        # shape is ?, assign best anchor to each gt
        best_prior_indices = anchor_ious.argmax(1)
        best_box_indices = (
            best_prior_indices * grid_x * grid_y
            + gt_batch[:, 3] * grid_x
            + gt_batch[:, 2]
        ).long()

        # 2. convert prediction to xyxy

        # gather dx, dy, w, h
        # in shape of (?, 4)
        # convert (dx, dy) to (cx, cy)
        # convert gt into cxcywh
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
                self.anchors * pred_batch[:, :, 2:4].exp(),
            ),
            2,
        )
        # B*A*H*W, 4
        preds = preds.permute(0, 1, 3, 4, 2).reshape(-1, 4)
        preds = box_convert(preds, "cxcywh", "xyxy")

        num_boxes = grid_x * grid_y * self.num_anchors
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
            best_gt_overlap, best_gt_idx = pred_iou.max(0)
            # shape is (A*H*W,), only ? is true
            over_threshold = best_gt_overlap > self.iou_threshold
            # shape is (?,)
            positivity[over_threshold] = 2

            # num_gt, 7
            gt = gt_batch[offset : offset + span]
            best_prior_idx = best_prior_indices[offset : offset + span]
            best_box_idx = best_box_indices[offset : offset + span]
            positivity[best_box_idx] = 1

            # ?, 7
            # C is dx, dy, w, h, class, iou, best_box_idx
            placeholder = torch.zeros(span, 7).to(self.device)
            placeholder[:, :4] = self.encode(
                gt[:, [0, 1, 4, 5]],
                self.anchors[0, best_prior_idx, :, 0, 0],
            )
            placeholder[:, 4] = gt[:, 6]
            placeholder[:, 5] = pred_iou[
                torch.arange(span).to(self.device),
                best_box_idx,
            ]

            placeholder[:, 6] = best_box_idx + i * num_boxes

            # dedup
            # if span > 1:
            #     visited = set()
            #     first_gt_indices = []
            #     for i, v in enumerate(best_box_idx.tolist()):
            #         if v not in visited:
            #             first_gt_indices.append(i)

            #     placeholder = placeholder[first_gt_indices]

            target.append(placeholder)
            positivities.append(positivity)

            offset += span

        # ?, 7 # B, A*H*W
        return target, torch.stack(positivities)

    def calc_prior_loss(
        self,
        pred: torch.Tensor,
        indicator: torch.Tensor,
        seen: int,
        grid_x: int,
        grid_y: int,
    ) -> torch.Tensor:
        prior_loss = 0
        if seen < self.prior_loss_threshold:
            anchors_truth = torch.zeros_like(pred)
            anchors_truth[:, 0] = 0.5 / grid_x
            anchors_truth[:, 1] = 0.5 / grid_y

            prior_loss = F.mse_loss(
                pred[indicator],
                anchors_truth[indicator],
                reduction="sum",
            )

        return prior_loss

    def forward(
        self,
        pred: torch.Tensor,
        gt: list[list[list[int]]],
        seen: int = 0,
    ) -> torch.Tensor:
        """forward function of YOLOv2Loss
        Some extra rules
        positive anchors: x,y,w,h,c,p loss
        before 12800, negative anchors : use anchors as truths
        best matched, iou lower than threshold: noobject loss
        best matched, iou over threshold: no loss

        p.s. match with fixed anchor and no overlapping groundtruth(?

        Args:
            pred (torch.Tensor): prediction, in format of (B, A*(4+1+C), H, W)
            gt (list[list[list[int]]]): batch of groundtruth, in format of (cx, cy, w, h, c)
            seen (int, optional): had seen how many images, 12800 is the threshold in the paper, Default is 0.

        Returns:
            torch.Tensor: loss
        """
        _, channel, grid_y, grid_x = pred.shape
        pred = pred.unflatten(1, (self.num_anchors, channel // self.num_anchors))

        # transform
        pred[:, :, :2, :, :] = pred[:, :, :2, :, :].sigmoid()
        pred[:, :, 4, :, :] = pred[:, :, 4, :, :].sigmoid()
        pred[:, :, 5:, :, :] = pred[:, :, 5:, :, :].softmax(2)

        # shape is (B, 7), format is (dx, dy, grid_x, grid_y, w, h, c)
        gt, spans = build_flatten_targets(gt, (grid_y, grid_x), delta_coord=True)
        gt = gt.to(self.device)

        # target shape is (?, C=7)
        # C is dx, dy, w, h, class, iou, best_box_idx
        # positivity shape is B, A*H*W
        with torch.no_grad():
            target, positivity = self.match(gt, spans, pred, grid_x, grid_y)

        # ?
        target = torch.cat(target)
        # B*A*H*W
        positivity = positivity.view(-1)
        # B*A*H*W, 5+C
        pred = pred.permute(0, 1, 3, 4, 2).reshape(-1, 5 + self.num_classes)

        # B*A*H*W
        negative_mask = positivity.eq(0)
        # no object loss for lower than threshold
        noobj_loss = F.mse_loss(
            pred[negative_mask, 4],
            torch.zeros(negative_mask.sum()).to(self.device),
            reduction="sum",
        )

        # before 12800 iter, prior as truth
        prior_loss = self.calc_prior_loss(
            pred[:, :4],
            positivity.ne(1),
            seen,
            grid_x,
            grid_y,
        )

        # good predictors
        # class loss / objecness loss / xywh loss
        # positive_mask = positivity.eq(2)
        # best_mask = positivity.eq(1)
        best_indices = target[:, 6].long()
        coord_loss = F.mse_loss(
            pred[best_indices, :4],
            target[:, :4],
            reduction="sum",
        )
        obj_loss = F.mse_loss(
            pred[best_indices, 4],
            target[:, 5],
            reduction="sum",
        )
        cls_loss = F.mse_loss(
            pred[best_indices, 5:],
            F.one_hot(target[:, 4].long(), self.num_classes).float(),
            reduction="sum",
        )

        total_loss = (
            cls_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_obj * obj_loss
            + self.lambda_coord * coord_loss
            + self.lambda_prior * prior_loss
        )

        if self.return_all_loss:
            return total_loss, cls_loss, noobj_loss, obj_loss, coord_loss
        else:
            return total_loss
