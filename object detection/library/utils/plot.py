import os
import random
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import lru_cache

def rand_color() -> List[int]:
    return random.choices(range(256), k=3)

@lru_cache
def load_image(path: str) -> np.ndarray:
    assert os.path.exists(path), f"Image {path} does't exist."
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def tensor2np_uint8(img: torch.Tensor) -> np.ndarray:
    assert img.dim() == 3, "please provide tensor in shape C,H,W."
    return (img.permute(1, 2, 0).numpy().copy() * 255).astype(np.uint8)


# labels
def draw_label(img: np.ndarray, label: torch.Tensor, width: int, height: int):
    """draw detection label"""

    for x, y, w, h, c in label:
        x, y, w, h = x * width, y * height, w * width, h * height
        x1, y1, x2, y2 = (
            int(x - w / 2),
            int(y - h / 2),
            int(x + w / 2),
            int(y + h / 2),
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_pred(
    img: np.ndarray,
    bboxes: torch.Tensor,
    class_name: List[str],
    class_show: bool = True,
    verbose: bool = True,
    class_color: Dict[str, List[int]] = None,
):
    """draw predicted bbox"""

    for bbox in bboxes:
        x, y, w, h, _ = bbox[:5]
        x, y, w, h = int(x), int(y), int(w), int(h)
        prob, cls_index = bbox[5:].max(0)
        prob = prob.numpy().round(2)
        cls_name = class_name[cls_index.item()]
        class_info = f"{cls_name}: {prob :.2f}"
        if verbose:
            print(class_info)

        color = class_color.get(cls_name, (0, 0, 255))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        if class_show:
            cv2.rectangle(img, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(
                img,
                text=class_info,
                org=(x + 5, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=1,
            )


def show_anchors(anchor_path: str):
    with open(anchor_path, "r") as f:
        wh = [l.strip().split(",") for l in f.readlines()]

    canvas = np.zeros((100, 100))

    for w, h in wh:
        w, h = float(w), float(h)
        x1, y1, x2, y2 = (
            100 * (0.5 - w / 2),
            100 * (0.5 - h / 2),
            100 * (0.5 + w / 2),
            100 * (0.5 + h / 2),
        )
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)

    plt.imshow(canvas)
    plt.show()
