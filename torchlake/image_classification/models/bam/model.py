from ..resnet.model import ConvBlockArgs, ResNet
from .network import BAM2d


class BAMResNet(ResNet):

    def build_blocks(self, **kwargs):
        """place BAM at each bottleneck of the network [1807.06514v2],
        i.e. at the end of every stage except the last, before the next stage downsamples
        """
        super().build_blocks(**kwargs)

        for stage, stage_cfg in zip(self.blocks[:-1], self.config):
            if isinstance(stage_cfg, ConvBlockArgs):
                output_channel = stage_cfg.output_dim
            else:
                output_channel = stage_cfg.hidden_dim * stage_cfg.expansion

            stage.append(BAM2d(output_channel))
