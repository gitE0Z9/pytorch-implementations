from itertools import product
from math import sqrt
from pathlib import Path
import numpy as np

import torch


class PriorBox:
    def __init__(self):
        self.feature_sizes = [38, 19, 10, 5, 3, 1]
        self.min_scale = 0.2
        self.max_scale = 0.9
        self.num_anchors = [4, 6, 6, 6, 4, 4]
        self.aspect_ratios = [1, 2, 1 / 2, 3, 1 / 3]
        self.anchors_path = f"configs/ssd/anchors.txt"

        if Path(self.anchors_path).exists():
            self.anchors = self.load_anchors()

    def load_anchors(self) -> torch.Tensor:
        anchors = np.loadtxt(self.anchors_path, delimiter=",")
        anchors = torch.from_numpy(anchors).float()

        return anchors

    def build_anchors(self) -> torch.Tensor:
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

    def save_anchors(self, anchors: np.ndarray):
        with open(self.anchors_path, "w") as f:
            for x, y, w, h in anchors.tolist():
                print(f"{x},{y},{w},{h}", file=f)
