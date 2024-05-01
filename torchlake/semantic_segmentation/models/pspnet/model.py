import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchlake.common.network import ConvBnRelu
from .network import PyramidPool2d


class PspNet(nn.Module):

    def __init__(
        self,
        latent_dim: int = 1,
        num_class: int = 1,
        bins_size: list[int] = [1, 2, 3, 6],
        dropout: float = 0.5,
        resent_no: 18 | 34 | 50 | 101 | 152 = 50,
    ):
        """Pyramid spatial pooling network [1612.01105v2]

        Args:
            num_class (int, optional): number of class. Defaults to 1.
            backbone (nn.Module, optional): backbone. Defaults to resnet50().
        """
        super(PspNet, self).__init__()
        self.load_backbone(resent_no)
        self.psp_layer = PyramidPool2d(latent_dim, bins_size)
        self.fc = nn.Sequential(
            ConvBnRelu(latent_dim * 2, 512, 3, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, num_class, 1),
        )

        if self.training:
            self.aux = nn.Sequential(
                ConvBnRelu(1024, 256, 3, padding=1),
                nn.Dropout2d(p=dropout),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        if self.training:
            aux, y = features
        else:
            y = features[0]
        y = self.psp_layer(y)
        y = self.fc(y)
        y = F.interpolate(y, x.shape[2:], mode="bilinear")

        if self.training:
            aux = self.aux(aux)
            aux = F.interpolate(aux, x.shape[2:], mode="bilinear")
            return y, aux

        return y
