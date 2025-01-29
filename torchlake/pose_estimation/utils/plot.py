from typing import Sequence

import cv2
import numpy as np
import torch


def draw_skeleton(
    img: np.ndarray,
    keypoints: torch.Tensor,
    masks: torch.Tensor | None = None,
    class_names: Sequence[str] | None = None,
    class_edges: Sequence[tuple[str, str]] | None = None,
    class_colors: dict[str, list[int]] = {},
    class_show: bool = True,
    radius: int = 8,
):
    """draw pose estimation label

    Args:
        img (np.ndarray): image
        keypoints (torch.Tensor): joints, in shape (num_person, num_joints, 2).
        masks (torch.Tensor | None, optional): masks, in shape (num_person, num_joints, 1). Defaults to None.
        class_names (Sequence[str] | None, optional): class names. Defaults to None.
        class_edges (Sequence[tuple[str, str]] | None, optional): class edges. Defaults to None.
        class_colors (Dict[str, List[int]], optional): palette for each class. Defaults to {}.
        class_show (bool, optional): show class and score. Defaults to True.
        radius (int, optional): radius of point mark. Defaults to 8.
    """
    FALLBACK_COLOR = (0, 0, 255)

    for p_idx, points in enumerate(keypoints):
        # point
        for c, (x, y) in enumerate(points):
            class_name = class_names[c]
            color = class_colors.get(class_name, FALLBACK_COLOR)

            if masks is not None and masks[p_idx, c].item():
                continue

            cv2.circle(img, (int(x), int(y)), radius, color, -1)

            # y1 = max(y1 - 30, 0)
            # y2 = y1 if y1 > 0 else y1 + 40
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # tx = x1 + 5
            # ty = y1 - 10 if y1 > 0 else y1 + 35
            # class_info = f"{class_name}"
            # cv2.putText(
            #     img,
            #     text=class_info,
            #     org=(tx, ty),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.8,
            #     color=(255, 255, 255),
            #     thickness=1,
            # )

        # edge
        if not class_edges:
            continue

        for s, e in class_edges:
            if masks is not None and (masks[p_idx, s].item() or masks[p_idx, e].item()):
                continue

            sx, sy = points[s]
            ex, ey = points[e]
            cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), FALLBACK_COLOR, 2)
