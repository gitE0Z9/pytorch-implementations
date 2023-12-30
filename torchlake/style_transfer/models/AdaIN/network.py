import torch
from torch import nn
from torchlake.common.utils.numerical import safe_std


class AdaIn2d(nn.Module):
    def __init__(self, content: torch.Tensor, style: torch.Tensor):
        super(AdaIn2d, self).__init__()
        self.mu_content = content.mean((2, 3), keepdim=True)
        self.beta_style = style.mean((2, 3), keepdim=True)

        self.sigma_content = safe_std(content, (2, 3), keepdim=True)
        self.gamma_style = safe_std(style, (2, 3), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sd = (x - self.mu_content) / self.sigma_content
        return self.gamma_style * sd + self.beta_style


# mirror encoder
class AdaInDecoderBlock(nn.Module):
    def __init__(
        self,
        layer_number: int,
        input_ch: int,
        output_ch: int,
        upsample: bool = True,
        drop_channel_first: bool = False,
    ):
        super(AdaInDecoderBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                input_ch,
                input_ch // 2 if drop_channel_first else input_ch,
                3,
                1,
                0,
            ),
            nn.ReLU(inplace=True),
        ]
        for i in range(1, layer_number):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(
                nn.Conv2d(
                    input_ch,
                    output_ch if i + 1 == layer_number else input_ch,
                    3,
                    1,
                    0,
                )
            )
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.upsample = (
            nn.UpsamplingNearest2d(scale_factor=2) if upsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        y = self.upsample(y)

        return y


class AdaInDecoder(nn.Module):
    def __init__(self):
        super(AdaInDecoder, self).__init__()
        self.block1 = AdaInDecoderBlock(1, 512, 256, drop_channel_first=True)
        self.block2 = AdaInDecoderBlock(4, 256, 128)
        self.block3 = AdaInDecoderBlock(2, 128, 64)
        self.block4 = AdaInDecoderBlock(2, 64, 3, upsample=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        return y
