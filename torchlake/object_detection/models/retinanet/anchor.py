from itertools import product
from math import sqrt

import torch


class PriorBox:
    def __init__(
        self,
        feature_sizes=[75, 38, 19, 10, 10],
        scales=[2**0, 2 ** (1 / 3), 2 ** (2 / 3)],
        num_anchors=9,
        aspect_ratios=[1 / 2, 1, 2],
    ):
        """prior box of RetinaNet"""
        self.feature_sizes = feature_sizes
        self.scales = scales
        self.num_anchors = num_anchors
        self.aspect_ratios = aspect_ratios

    def build_anchors(self) -> torch.Tensor:
        """build anchors

        Returns:
            torch.Tensor: anchors, in format of (cx, cy, w, h), in shape of (#num_anchor * #num_grid_y * #num_grid_x, 4)
        """
        anchors = []
        for feature_size in self.feature_sizes:
            for i, j in product(range(feature_size), repeat=2):  # mesh grid y,x index
                cx = (j + 0.5) / feature_size
                cy = (i + 0.5) / feature_size

                for ar, scale in product(self.aspect_ratios, self.scales):
                    anchors.append(
                        [
                            cx,
                            cy,
                            (1 / feature_size) * sqrt(ar) * scale,
                            (1 / feature_size) / sqrt(ar) * scale,
                        ]
                    )

        anchors = torch.Tensor(anchors)
        anchors.clamp_(0, 1)
        return anchors
