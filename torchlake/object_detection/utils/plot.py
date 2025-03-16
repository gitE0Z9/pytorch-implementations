from typing import Sequence
import cv2
import numpy as np
import torch


def draw_label(
    img: np.ndarray,
    label: torch.Tensor,
    class_names: Sequence[str] | None = None,
    class_colors: dict[str, list[int]] = {},
    class_show: bool = True,
):
    """draw detection label

    Args:
        img (np.ndarray): image
        label (torch.Tensor): normalized x,y,w,h coordinates and class index, in shape (N, 5) and format (x, y, w, h, c)
        class_names (Sequence[str] | None, optional): class names. Defaults to None.
        class_colors (Dict[str, List[int]], optional): palette for each class. Defaults to {}.
        class_show (bool, optional): show class and score. Defaults to True.
    """
    FALLBACK_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    height, width, _ = img.shape

    for x, y, w, h, c in label:
        x, y, w, h = x * width, y * height, w * width, h * height
        x1, y1, x2, y2 = (
            int(x),
            int(y),
            int(x + w),
            int(y + h),
        )
        class_name = class_names[int(c)] if class_names else None
        color = class_colors.get(class_name, FALLBACK_COLOR)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if class_show:
            x1 = x1
            x2 = int(x1 + w)
            y1 = max(y1 - 30, 0)
            y2 = y1 if y1 > 0 else y1 + 40
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            tx = x1 + 5
            ty = y1 - 10 if y1 > 0 else y1 + 35
            class_info = f"{class_name}"
            cv2.putText(
                img,
                text=class_info,
                org=(tx, ty),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=TEXT_COLOR,
                thickness=1,
            )


def draw_pred(
    img: np.ndarray,
    bboxes: torch.Tensor,
    class_names: list[str],
    class_colors: dict[str, list[int]] = {},
    class_show: bool = True,
    verbose: bool = True,
):
    """draw predicted bbox

    Args:
        img (np.ndarray): image
        bboxes (torch.Tensor): decoded bounding boxes, in shape (N, 5+C)
        class_name (List[str]): class names
        class_color (Dict[str, List[int]], optional): palette for each class. Defaults to {}.
        class_show (bool, optional): show class and score. Defaults to True.
        verbose (bool, optional): print out class and score. Defaults to True.
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


def draw_anchors(anchors: torch.Tensor) -> np.ndarray:
    """draw anchors

    Args:
        anchors (torch.Tensor): normalized anchors, in shape (N, 4), in format (cx, cy, w, h)

    Returns:
        canvas: image, in shape (100, 100)
    """
    BORDER_COLOR = (255, 255, 255)
    SCALE = 100

    wh = anchors[:, -2:].float()
    xyxy = (SCALE * torch.cat([0.5 - wh / 2, 0.5 + wh / 2], 1)).int()

    canvas = np.zeros((SCALE, SCALE))
    for x1, y1, x2, y2 in xyxy.tolist():
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER_COLOR, 1)

    return canvas
