import enum
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..datasets.mpii import MPIIFromRaw


def test_mpii_from_raw():
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(0, 1),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    dataset = MPIIFromRaw(
        Path(__file__)
        .parent.parent.parent.parent.parent.joinpath("data")
        .joinpath("MPII"),
        mode="train",
        transform=transform,
    )

    # check shape
    assert len(dataset) > 1
    assert dataset[0][0].shape == torch.Size((3, 256, 256))
    assert dataset[0][1].shape == torch.Size((16, 2))

    # run through whole dataset
    # for i in range(len(dataset)):
    #     dataset[i]
