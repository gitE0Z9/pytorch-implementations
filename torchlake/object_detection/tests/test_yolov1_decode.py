import torch

from torch.testing import assert_close
from ..models.yolov1.decode import Decoder, yolo_postprocess
from ..configs.schema import InferenceCfg
from ..constants.schema import DetectorContext

NUM_CLASS = 2
NUM_ANCHOR = 2
INFERENCE_CFG = InferenceCfg(METHOD="torchvision", CONF_THRESH=0.3, NMS_THRESH=0.5)
DETECTOR_CONTEXT = DetectorContext(
    detector_name="yolov1",
    dataset="voc",
    device="cpu",
    num_classes=NUM_CLASS,
    num_anchors=NUM_ANCHOR,
)


def test_yolo_postprocess_output():
    # shape: 2, 2*1*1, 5+2
    decoded = torch.Tensor(
        [
            # this batch has same position and same class
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.7],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
            ],
            # this batch has same position and diff class
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
            ],
        ],
    )

    output = yolo_postprocess(decoded, NUM_CLASS, INFERENCE_CFG)

    expected = [
        torch.Tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.16, 0.64],
            ]
        ),
        torch.Tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.35, 0.15],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.16, 0.64],
            ]
        ),
    ]

    assert_close(output, expected)


def test_decoder_shape():
    decoder = Decoder(DETECTOR_CONTEXT)

    # shape:
    feature_map = torch.Tensor(
        [
            # this batch has same position and same class
            [
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
            ],
            # this batch has same position and diff class
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.3],
                [0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8],
            ],
        ]
    )

    output = decoder.decode(feature_map, [224, 224])

    expected = torch.Tensor(
        [
            # this batch has same position and same class
            [
                [112, 112, 112, 112, 0.5, 0.3, 0.7],
                [112, 112, 112, 112, 0.8, 0.2, 0.8],
            ],
            # this batch has same position and diff class
            [
                [112, 112, 112, 112, 0.5, 0.7, 0.3],
                [112, 112, 112, 112, 0.8, 0.2, 0.8],
            ],
        ],
    )

    assert_close(output, expected)
