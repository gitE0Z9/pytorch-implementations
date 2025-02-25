import torch
import torch.nn.functional as F
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase


class NeuralDoodle(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        layer_names: list[str],
        mask_weight: float = 1,
    ):
        super().__init__(None, None, foot_kwargs={"backbone": backbone})
        self.layer_names = layer_names
        self.mask_weight = mask_weight

    #   self.neighbor_conv = [nn.Conv2d(i+3,3,3,bias=False).to(device) for i in [256,512]]

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("backbone")

    def build_head(self, _, **kwargs):
        pass

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        features = self.foot(img, self.layer_names)
        for layer_name, feature in zip(self.layer_names, features):
            block_num, _ = layer_name.split("_")
            downscale = 2 ** (int(block_num) - 1)
            pooling = F.avg_pool2d(mask, downscale)

            try:
                feature = torch.cat(
                    [
                        feature,
                        self.mask_weight
                        * F.interpolate(pooling, size=feature.shape[2:]),
                    ],
                    dim=1,
                )
            except:
                print(feature.shape, pooling.shape)
                raise ValueError

        return features

        # for a, c in zip(aux_list, self.neighber_conv):
        #     a = c(a)
