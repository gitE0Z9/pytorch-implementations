import matplotlib.pyplot as plt
import numpy as np


def plot_grids(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    num_row: int = 1,
    num_col: int = 1,
    figsize: tuple[int] = (8, 8),
):
    plt.figure(figsize=figsize)

    for idx, image in enumerate(images):
        plt.subplot(num_row, num_col, idx + 1)
        if titles:
            title = titles[idx]
            plt.title(title)
        plt.axis("off")
        plt.imshow(image)
