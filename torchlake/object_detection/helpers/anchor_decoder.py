from typing import Sequence

import torch

from ..configs.schema import InferenceCfg
from ..constants.schema import DetectorContext
from ..utils.config import load_anchors
from ..utils.nms import select_best_index


class Decoder:
    def __init__(
        self,
        context: DetectorContext,
        anchors: torch.Tensor = None,
    ):
        self.anchors = anchors
        if anchors is None:
            self.anchors = load_anchors(context.anchors_path)

        self.context = context

    def decode(
        self,
        pred: torch.Tensor,
        image_size: Sequence[int],
    ) -> torch.Tensor:
        """decode output to detections

        Args:
            pred (torch.Tensor): pred, in shape of (batch size, #num_anchor * #num_grid_y * #num_grid_x, 4 + 1 + num_class)
            image_size (Sequence[int]): image height, image width

        Returns:
            torch.Tensor: decoded output, in shape of (batch size, #num_anchor * #num_grid_y * #num_grid_x, 4+num_class+1)
        """
        loc_pred, conf_pred = pred[:, :, :4], pred[:, :, 4:]
        input_h, input_w = image_size

        # cxcy: yhat * anchors + anchors
        # wh: yhat.exp() * anchors
        loc_pred[:, :, :2] = (
            loc_pred[:, :, :2] * self.anchors[:, 2:4] + self.anchors[:, :2]
        )
        loc_pred[:, :, 2:] = loc_pred[:, :, 2:].exp() * self.anchors[:, 2:4]
        loc_pred[:, :, 0] *= input_w
        loc_pred[:, :, 1] *= input_h
        loc_pred[:, :, 2] *= input_w
        loc_pred[:, :, 3] *= input_h

        # cx, cy -> x, y
        loc_pred[:, :, :2] -= loc_pred[:, :, 2:] / 2

        conf_pred = conf_pred.float().softmax(dim=-1)

        # batch size, #num_anchor * #num_grid_y * #num_grid_x, 4+num_class+1
        return torch.cat([loc_pred, conf_pred], -1)

    def post_process(
        self,
        decoded: torch.Tensor,
        postprocess_config: InferenceCfg,
    ) -> list[torch.Tensor]:
        """post process yolo decoded output

        Args:
            decoded (torch.Tensor): shape (#batch, #anchor * #grid, 5 + #class)
            postprocess_config (InferenceCfg): post process config for nms

        Returns:
            list[torch.Tensor]: #batch * (#selected, 5 + #class)
        """
        batch_size, _, _ = decoded.shape
        cls_indices = decoded[:, :, 4:].argmax(-1)

        processed_result = []
        for batch_idx in range(batch_size):
            detection_result = []
            # plus 1 to skip background class
            for class_index in range(1, self.context.num_classes + 1):
                is_this_class_candidate: torch.Tensor = cls_indices[batch_idx].eq(
                    class_index
                )
                if not is_this_class_candidate.any():
                    continue

                this_class_detection = decoded[batch_idx, is_this_class_candidate]
                best_index = select_best_index(
                    this_class_detection[:, :4],
                    this_class_detection[:, 4 + class_index],
                    postprocess_config,
                )
                detection_result.append(this_class_detection[best_index])

            # for broken detector
            if len(detection_result) == 0:
                processed_result.append([])
            else:
                processed_result.append(torch.cat(detection_result, 0))

        return processed_result
