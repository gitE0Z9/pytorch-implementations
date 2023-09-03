import os
import random
from typing import Dict, List

import cv2
import numpy as np
import torch


def rand_color() -> List[int]:
    return random.choices(range(256), k=3)


def load_image(path: str) -> np.ndarray:
    assert os.path.exists(path), "Image does't exist."
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def tensor2np_uint8(img: torch.Tensor) -> np.ndarray:
    assert img.dim() == 3, "please provide tensor in shape C,H,W."
    return (img.permute(1, 2, 0).numpy().copy() * 255).astype(np.uint8)


# labels
def draw_label(img: np.ndarray, label: torch.Tensor, width: int, height: int):
    """draw detection label"""

    for i, (x, y, w, h, c) in enumerate(label):
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
                class_info,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
            )
