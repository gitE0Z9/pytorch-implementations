from pathlib import Path

import albumentations as A
import pytest
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..datasets.mpii import MPIIFromRaw


@pytest.mark.parametrize("crop_by_person", [True, False])
def test_mpii_from_raw(crop_by_person):
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
        crop_by_person=crop_by_person,
        transform=transform,
    )

    # check shape
    assert len(dataset) > 1
    img, label = dataset[0]
    num_person = 2
    if crop_by_person:
        assert img.shape == torch.Size((num_person, 3, 256, 256))
    else:
        assert img.shape == torch.Size((3, 256, 256))

    assert label.shape == torch.Size((num_person, 16, 3))

    # run through whole dataset
    # for i in range(len(dataset)):
    #     dataset[i]
