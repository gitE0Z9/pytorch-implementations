import cv2
import numpy as np
import torch


def draw_label(
    img: np.ndarray,
    label: torch.Tensor,
    class_names: list[str],
    class_show: bool = True,
    class_colors: dict[str, list[int]] = {},
):
    """draw pose estimation label

    Args:
        img (np.ndarray): image
        label (torch.Tensor): joints, in shape (num_person, num_joints, 2)
        class_name (List[str]): class names
        class_show (bool, optional): show class and score. Defaults to True.
        verbose (bool, optional): print out class and score. Defaults to True.
        class_color (Dict[str, List[int]], optional): palette for each class. Defaults to {}.
    """
    height, width, _ = img.shape

    for p_idx, points in enumerate(label):
        for c, (x, y) in enumerate(points):
            class_name = class_names[c]
            class_info = f"{class_name}"
            color = class_colors.get(class_name, (0, 0, 255))

            cv2.circle(img, (int(x), int(y)), 8, color, -1)

            # if class_show:
            # y1 = max(y1 - 30, 0)
            # y2 = y1 if y1 > 0 else y1 + 40
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # tx = x1 + 5
            # ty = y1 - 10 if y1 > 0 else y1 + 35
            # cv2.putText(
            #     img,
            #     text=class_info,
            #     org=(tx, ty),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.8,
            #     color=(255, 255, 255),
            #     thickness=1,
            # )
