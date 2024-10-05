from .fcn.model import FCN
from .unet.model import UNet
from .pspnet.model import PspNet
from .dual_attention.model import DaNet
from .deeplab.model import DeepLab

__all__ = [
    "FCN",
    "UNet",
    "PspNet",
    "DaNet",
    "DeepLab",
]
