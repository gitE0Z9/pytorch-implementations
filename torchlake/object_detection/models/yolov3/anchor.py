from pathlib import Path
from typing import Sequence

import torch

from ...constants.schema import DetectorContext
from ...utils.train import generate_grid_train
from ..yolov2.anchor import PriorBox


class PriorBox(PriorBox):
    def __init__(self, context: DetectorContext):
        assert isinstance(
            context.num_anchors, Sequence
        ), "number of anchors in context should be an array of integer"

        self.anchors_path = Path(context.anchors_path)
        self.num_anchors = sum(context.num_anchors)
        self.context = context

    def anchors_to_cxcywh(
        self,
        anchors: torch.Tensor,
        grid_sizes: Sequence[int | Sequence[int]],
    ):
        num_anchors_in_multiscale = self.context.num_anchors
        assert len(num_anchors_in_multiscale) == len(
            grid_sizes
        ), "length of num_anchors and grid_sizes of a detector context should be the same"

        outputs = []
        offset = 0
        for grid_size, num_anchor_in_single_scale in zip(
            grid_sizes, num_anchors_in_multiscale
        ):
            if isinstance(grid_size, int):
                grid_size = (grid_size, grid_size)

            grid = generate_grid_train(*grid_size, is_center=True).to(anchors.device)
            _, _, _, h, w = grid.shape

            # 1, A*H*W, 2
            grid = (
                grid.repeat(1, num_anchor_in_single_scale, 1, 1, 1)
                .permute(0, 1, 3, 4, 2)
                .flatten(1, 3)
            )

            # 1, A*H*W, 2
            anchors_in_single_scale = (
                anchors[:, offset : offset + num_anchor_in_single_scale]
                .repeat(1, 1, 1, h, w)
                .permute(0, 1, 3, 4, 2)
                .flatten(1, 3)
            )

            # 1, A*H*W, 4
            outputs.append(torch.cat((grid, anchors_in_single_scale), -1))

            offset += num_anchor_in_single_scale

        # 1, num_boxes, 4
        return torch.cat(outputs, 1)
