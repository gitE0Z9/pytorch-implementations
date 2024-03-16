import random
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def rand_color() -> List[int]:
    return random.choices(range(256), k=3)


def tensor2np_uint8(img: torch.Tensor) -> np.ndarray:
    assert img.dim() == 3, "please provide tensor in shape C,H,W."
    return (img.permute(1, 2, 0).numpy().copy() * 255).astype(np.uint8)


# labels
def draw_label(
    img: np.ndarray,
    label: torch.Tensor,
    class_names: List[str],
    class_show: bool = True,
    class_colors: Dict[str, List[int]] = None,
):
    """draw detection label

    Args:
        img (np.ndarray): image
        label (torch.Tensor): scaled x,y,w,h coordinates and class index
        class_name (List[str]): class names
        class_show (bool, optional): show class and score. Defaults to True.
        verbose (bool, optional): print out class and score. Defaults to True.
        class_color (Dict[str, List[int]], optional): palette for each class. Defaults to None.
    """
    height, width, _ = img.shape

    for x, y, w, h, c in label:
        x, y, w, h = x * width, y * height, w * width, h * height
        x1, y1, x2, y2 = (
            int(x - w / 2),
            int(y - h / 2),
            int(x + w / 2),
            int(y + h / 2),
        )
        class_name = class_names[c]
        class_info = f"{class_name}"
        color = class_colors.get(class_name, (0, 0, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if class_show:
            x1 = x1
            x2 = int(x1 + w)
            y1 = max(y1 - 30, 0)
            y2 = y1 if y1 > 0 else y1 + 40
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            tx = x1 + 5
            ty = y1 - 10 if y1 > 0 else y1 + 35
            cv2.putText(
                img,
                text=class_info,
                org=(tx, ty),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=1,
            )


def draw_pred(
    img: np.ndarray,
    bboxes: torch.Tensor,
    class_names: List[str],
    class_show: bool = True,
    verbose: bool = True,
    class_colors: Dict[str, List[int]] = None,
):
    """draw predicted bbox

    Args:
        img (np.ndarray): image
        bboxes (torch.Tensor): decoded bounding boxes
        class_name (List[str]): class names
        class_show (bool, optional): show class and score. Defaults to True.
        verbose (bool, optional): print out class and score. Defaults to True.
        class_color (Dict[str, List[int]], optional): palette for each class. Defaults to None.
    """

    for bbox in bboxes:
        x, y, w, h, _ = bbox[:5]
        x, y, w, h = int(x), int(y), int(w), int(h)
        prob, cls_index = bbox[5:].max(0)
        prob = prob.numpy().round(2)
        class_name = class_names[cls_index.item()]
        class_info = f"{class_name}: {prob :.2f}"
        if verbose:
            print(class_info)

        color = class_colors.get(class_name, (0, 0, 255))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        if class_show:
            x1 = x
            x2 = x + w
            y1 = max(y - 30, 0)
            y2 = y if y1 > 0 else y + 40
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            tx = x + 5
            ty = y - 10 if y1 > 0 else y + 35
            cv2.putText(
                img,
                text=class_info,
                org=(tx, ty),
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
