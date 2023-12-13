import torch
import torch.nn.functional as F
from torch import nn
from ..base.network import FeatureExtractor


class AuxiliaryNetwork(nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        layer_names: list[str],
        mask_weight: float = 1,
    ):
        super(AuxiliaryNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.layer_names = layer_names
        self.mask_weight = mask_weight

    #   self.neighbor_conv = [nn.Conv2d(i+3,3,3,bias=False).to(device) for i in [256,512]]

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor]]:
        features = self.feature_extractor(img, self.layer_names)
        for layer_name, feature in zip(self.layer_names, features):
            block_num, _ = layer_name.split("_")
            downscale = 2 ** (int(block_num) - 1)
            pooling = F.avg_pool2d(mask, downscale)

            try:
                feature = torch.cat(
                    [
                        feature,
                        self.mask_weight
                        * pooling[:, :, : feature.size(2), : feature.size(3)],
                    ],
                    dim=1,
                )
            except:
                print(feature.shape, pooling.shape)
                raise ValueError

        return features

        # for a, c in zip(aux_list, self.neighber_conv):
        #     a = c(a)
