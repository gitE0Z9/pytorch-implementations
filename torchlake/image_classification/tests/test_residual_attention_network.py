import pytest
import torch

from ..models.resnet.network import ResBlock
from ..models.residual_attention.model import ResidualAttentionNetwork
from ..models.residual_attention.network import MaskBranch, AttentionModule


class TestResidualAttentionNetwork:
    @pytest.mark.parametrize(
        "name,input_channel,output_channel,num_skip",
        [
            ["first-skip-2", 64, 256, 2],
            ["middle-skip-2", 256, 256, 2],
            ["first-skip-1", 64, 256, 1],
            ["middle-skip-1", 256, 256, 1],
            ["first-skip-0", 64, 256, 0],
            ["middle-skip-0", 256, 256, 0],
        ],
    )
    def test_mask_branch_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        num_skip: int,
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = MaskBranch(input_channel, output_channel, num_skip=num_skip)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, 13, 13))

    @pytest.mark.parametrize(
        "name,input_channel,output_channel",
        [
            ["bottleneck_first", 64, 256],
            ["bottleneck_middle", 256, 256],
        ],
    )
    def test_attention_module_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
    ):
        x = torch.randn(2, input_channel, 13, 13)
        layer = AttentionModule(input_channel, output_channel)
        y = layer(x)

        assert y.shape == torch.Size((2, output_channel, 13, 13))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["56", 56],
            ["92", 92],
        ],
    )
    def test_residual_attention_network_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = ResidualAttentionNetwork(output_size=5, num_layer=num_layer)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
