from itertools import product
from math import sqrt

import torch


class PriorBox:
    def __init__(
        self,
        feature_sizes=[38, 19, 10, 5, 3, 1],
        min_scale=0.2,
        max_scale=0.9,
        num_anchors=[4, 6, 6, 6, 4, 4],
        aspect_ratios=[1, 2, 1 / 2, 3, 1 / 3],
    ):
        """prior box of SSD"""
        self.feature_sizes = feature_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_anchors = num_anchors
        self.aspect_ratios = aspect_ratios

    def build_anchors(self) -> torch.Tensor:
        """build anchors

        Returns:
            torch.Tensor: anchors, in format of (cx, cy, w, h), in shape of (#num_anchor * #num_grid_y * #num_grid_x, 4)
        """
        anchors = []
        for k, feature_size in enumerate(self.feature_sizes):
            num_anchors = self.num_anchors[k]
            sk = self.min_scale + (self.max_scale - self.min_scale) * k / (
                len(self.feature_sizes) - 1
            )
            for i, j in product(range(feature_size), repeat=2):  # mesh grid y,x index
                cx = (j + 0.5) / feature_size
                cy = (i + 0.5) / feature_size

                for ar in self.aspect_ratios:
                    if (num_anchors == 4) and (ar in [3, 1 / 3]):
                        continue

                    anchors.append([cx, cy, sk * sqrt(ar), sk / sqrt(ar)])

                    if ar == 1:
                        sk_1 = self.min_scale + (self.max_scale - self.min_scale) * (
                            k + 1
                        ) / (len(self.feature_sizes) - 1)
                        sk_prime = sqrt(sk * sk_1)
                        # ar = 1, two anchors
                        anchors.append([cx, cy, sk_prime, sk_prime])

        anchors = torch.Tensor(anchors)
        anchors.clamp_(0, 1)
        return anchors
