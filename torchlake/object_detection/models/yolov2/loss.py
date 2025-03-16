import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlake.object_detection.constants.schema import DetectorContext
from torchlake.object_detection.models.yolov2.anchor import PriorBox
from torchlake.object_detection.utils.train import (
    build_flatten_targets,
    generate_grid_train,
)
from torchvision.ops import box_convert, box_iou


class YOLOV2Loss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        lambda_obj: float = 5,
        lambda_noobj: float = 1,
        lambda_prior: float = 0.01,
        lambda_coord: float = 1,
        iou_threshold: float = 0.6,
    ):
        super().__init__()

        self.num_anchors = context.num_anchors
        self.num_classes = context.num_classes
        self.device = context.device
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_prior = lambda_prior
        self.lambda_coord = lambda_coord
        self.iou_threshold = iou_threshold
        self.epsilon = 1e-5

        self.anchors = PriorBox(context).anchors.to(context.device)

    def encode(self, gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """encode gt loc information

        Args:
            gt (torch.Tensor): groundtruth coordinates in format of (cx, cy, w, h), shape is (?, 4)
            anchors (torch.Tensor): anchors, shape is (?, 4)

        Returns:
            torch.Tensor: target tensors, shape is (?, 4)
        """

        g_cxcy = gt[:, :2]
        g_wh = gt[:, 4:6].log() - anchors[:, 2:].log()

        # assert not torch.isnan(g_cxcy).any()
        # assert not torch.isnan(g_wh).any()

        return torch.cat([g_cxcy, g_wh], -1)

    def match(
        self,
        gt_batch: torch.Tensor,
        spans: torch.Tensor,
        grid_x: int,
        grid_y: int,
    ) -> torch.Tensor:
        """match anchor to groundtruth
        1. if iou over threshold, don't update obj/noobj loss
        2. before 12800 pictures, update prior loss
        3. update coord/obj/class loss for best box

        Args:
            gt_batch (torch.Tensor): batched bboxes, in format(dx, dy, grid_x, grid_y, w, h, c) if delta coord is true
            or in format of (cx, cy, w, h, c).
            spans (torch.Tensor): number of detections of each image
            grid_x (int): grid size of x dim
            grid_y (int): grid size of y dim

        Returns:
            torch.Tensor: in shape of (B, A, H, W, C=7), C is dx, dy, ln(w/a), ln(h/a), class, iou, positive_level
            positive_level includes 0:negative 1:object exist 2:positive
        """
        # anchors could be precomputed for static grid_x, grid_y under single scale scenario
        # A*H*W, 2
        grids = (
            generate_grid_train(grid_x, grid_y, center=True)
            .repeat(1, self.num_anchors, 1, 1, 1)
            .permute(0, 1, 3, 4, 2)
            .reshape(-1, 2)
            .to(self.device)
        )
        # A*H*W, 4
        anchors = torch.cat(
            [
                grids,
                self.anchors.repeat(1, 1, 1, grid_y, grid_x)
                .permute(0, 1, 3, 4, 2)
                .reshape(-1, 2),
            ],
            -1,
        )

        gts = torch.cat(
            [
                gt_batch[:, :2],
                gt_batch[:, 4:6],
            ],
            1,
        )
        gts[:, 0] += gt_batch[:, 2] / grid_x
        gts[:, 1] += gt_batch[:, 3] / grid_y
        # ?, A*H*W
        ious = box_iou(
            box_convert(gts, "cxcywh", "xyxy"),
            box_convert(anchors, "cxcywh", "xyxy"),
        )

        num_boxes = grid_x * grid_y * self.num_anchors
        target = []
        offset = 0
        for span in spans:
            # num_gt, A*H*W
            iou = ious[offset : offset + span]
            # num_gt, 5
            gt = gt_batch[offset : offset + span]

            # A*H*W, 7
            # C is cx, cy, w, h, class, conf, positive
            placeholder = torch.zeros(num_boxes, 7).to(self.device)

            # shape is (A*H*W,), assign gt to acceptable anchor
            best_gt_overlap, best_gt_idx = iou.max(0)
            # shape is (A*H*W,), only ? is true
            over_threshold = best_gt_overlap > self.iou_threshold
            # shape is (?,)
            gt_idx_of_acceptable_prior = best_gt_idx[over_threshold]
            # ?,6
            placeholder[over_threshold] = torch.cat(
                [
                    self.encode(
                        gt[gt_idx_of_acceptable_prior, :6],
                        anchors[over_threshold],
                    ),
                    gt[gt_idx_of_acceptable_prior, 6:7],
                    iou[gt_idx_of_acceptable_prior, over_threshold, None],
                    torch.full((gt_idx_of_acceptable_prior.size(0), 1), 2).to(
                        self.device
                    ),
                ],
                1,
            )

            # shape is (num_gt,), assign best anchor to each gt
            best_prior_idx = iou.argmax(1)

            # num_gt,6
            placeholder[best_prior_idx] = torch.cat(
                [
                    self.encode(gt[:, :6], anchors[best_prior_idx]),
                    gt[:, 6:7],
                    iou[torch.arange(span).to(self.device), best_prior_idx, None],
                    torch.ones(span, 1).to(self.device),
                ],
                1,
            )

            target.append(placeholder)

            offset += span

        # B, A*H*W, 7
        return torch.stack(target, 0)

    def forward(
        self,
        pred: torch.Tensor,
        gt: list[list[list[int]]],
        # seen: int,
    ) -> torch.Tensor:
        """forward function of YOLOv2Loss
        Some extra rules
        positive anchors: x,y,w,h,c,p loss
        before 12800, negative anchors : use anchors as truths
        best matched, iou lower than threshold: noobject loss
        best matched, iou over threshold: no loss

        p.s. match with fixed anchor and no overlapping groundtruth(?

        Args:
            prediction (torch.Tensor): prediction
            groundtruth (list): groundtruth
            anchors (torch.Tensor): anchors
            seen (int): had seen how many images, 12800 is the threshold in the paper

        Returns:
            torch.Tensor: loss
        """
        _, channel, grid_y, grid_x = pred.shape
        pred = pred.unflatten(1, (self.num_anchors, channel // self.num_anchors))

        # transform
        pred[:, :, 0:2, :, :] = pred[:, :, 0:2, :, :].sigmoid()
        # pred[:, :, 2:4, :, :] = pred[:, :, 2:4, :, :].exp() * self.anchors
        pred[:, :, 4, :, :] = pred[:, :, 4, :, :].sigmoid()
        pred[:, :, 5:, :, :] = pred[:, :, 5:, :, :].softmax(2)

        # shape is (B, 5), format is (cx, cy, w, h, p)
        gt, spans = build_flatten_targets(gt, (grid_y, grid_x), delta_coord=True)
        gt = gt.to(self.device)

        # shape is (B*A*H*W, C=7)
        # C is dx, dy, w, h, class, conf, positive
        target = self.match(gt, spans, grid_x, grid_y).reshape(-1, 7)
        # B*A*H*W, 5+C
        pred = pred.permute(0, 1, 3, 4, 2).reshape(-1, 5 + self.num_classes)

        # B, A, H, W
        negative_mask = target[:, 6].eq(0)
        # no object loss for lower than threshold
        noobj_loss = F.mse_loss(
            pred[negative_mask, 4],
            torch.zeros(negative_mask.sum()).to(self.device),
            reduction="sum",
        )

        # before 12800 iter, prior as truth
        # prior_loss = self.calc_prior_loss(
        #     pred,
        #     seen,
        #     batch_size,
        #     grid_y,
        #     grid_x,
        #     positive_indicator,
        # )

        # good predictors
        # class loss / objecness loss / xywh loss
        # positive_mask = target[:, 6].gt(0)
        best_mask = target[:, 6].eq(1)
        coord_loss = F.mse_loss(
            pred[best_mask, :4],
            target[best_mask, :4],
            reduction="sum",
        )
        obj_loss = F.mse_loss(
            pred[best_mask, 4],
            target[best_mask, 5],
            reduction="sum",
        )
        # cls_loss = F.cross_entropy(
        #     pred[best_mask, 5:],
        #     target[best_mask, 4].long(),
        #     reduction="sum",
        # )
        cls_loss = F.mse_loss(
            pred[best_mask, 5:],
            F.one_hot(target[best_mask, 4].long(), self.num_classes).float(),
            reduction="sum",
        )

        total_loss = (
            cls_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_obj * obj_loss
            + self.lambda_coord * coord_loss
            # + self.lambda_prior * prior_loss
        )

        return total_loss

    def calc_prior_loss(
        self,
        prediction: torch.Tensor,
        seen: int,
        batch_size: int,
        grid_y: int,
        grid_x: int,
        positive_indicator: torch.Tensor,
    ) -> torch.Tensor:
        prior_loss = 0
        if seen < 12800:
            # N, 5, 4, 13, 13
            box_pred = ~positive_indicator * prediction[:, :, 0:4, :, :]
            anchors_truth = torch.zeros_like(box_pred).to(self.device)
            anchors_truth[:, :, 0:1, :, :] = 0.5 / grid_x
            anchors_truth[:, :, 1:2, :, :] = 0.5 / grid_y
            anchors_truth[:, :, 2:4, :, :] = self.anchors.tile(
                batch_size, 1, 1, grid_y, grid_x
            )
            anchors_truth = ~positive_indicator * anchors_truth
            prior_loss = F.mse_loss(box_pred, anchors_truth, reduction="sum")

        return prior_loss


class YOLO9000Loss(nn.Module): ...
