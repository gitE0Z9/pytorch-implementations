import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops import Conv2dNormActivation

from .network import DualAttention2d


class DANet(nn.Module):

    def __init__(
        self,
        latent_dim: int = 1,
        num_class: int = 1,
        reduction_ratio: float = 32,
        dropout_prob: float = 0.1,
        resent_no: 18 | 34 | 50 | 101 | 152 = 50,
    ):
        """Dual Attention Network for Scene Segmentation [1809.02983v4]

        Args:
            latent_dim (int): dimension of latent representation
            num_class (int, optional): number of class. Defaults to 1.
        """
        super(DANet, self).__init__()
        self.load_backbone(resent_no)
        inter_channel = latent_dim // reduction_ratio
        self.att = DualAttention2d(latent_dim, reduction_ratio)
        self.head = nn.Sequential(
            nn.Dropout2d(dropout_prob, False),
            nn.Conv2d(inter_channel, num_class, 1),
        )

        if self.training:
            self.aux = nn.Sequential(
                Conv2dNormActivation(1024, 256),
                nn.Dropout2d(dropout_prob),
                nn.Conv2d(256, num_class, 1),
            )

    def load_backbone(
        self,
        resent_no: 18 | 34 | 50 | 101 | 152 = 50,
    ):
        self.cnn = getattr(torchvision.models, f"resnet{resent_no}", None)(
            weights="DEFAULT"
        )
        assert self.cnn, "resent_no not recognized"

        # dilation
        # memory hungry !!!
        # https://github.com/hszhao/semseg/blob/4f274c3f276778228bc14a4565822d46359f0cc8/model/pspnet.py#L49
        for key, layer in self.cnn.layer3.named_modules():
            if "conv2" in key:
                layer.dilation, layer.padding, layer.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in key:
                layer.stride = (1, 1)
        for key, layer in self.cnn.layer4.named_modules():
            if "conv2" in key:
                layer.dilation, layer.padding, layer.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in key:
                layer.stride = (1, 1)

    def get_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []

        y = self.cnn.conv1(x)
        y = self.cnn.bn1(y)
        y = self.cnn.relu(y)
        y = self.cnn.maxpool(y)
        y = self.cnn.layer1(y)
        y = self.cnn.layer2(y)
        y = self.cnn.layer3(y)
        if self.training:
            features.append(y)
        y = self.cnn.layer4(y)
        features.append(y)

        return features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor] | torch.Tensor:
        features = self.get_features(x)
        if self.training:
            aux, y = features
        else:
            y = features[0]
        y = self.att(y)
        y = self.head(y)
        y = F.interpolate(y, x.shape[2:], mode="bilinear")

        if self.training:
            aux = self.aux(aux)
            aux = F.interpolate(aux, x.shape[2:], mode="bilinear")
            return y, aux

        return y
