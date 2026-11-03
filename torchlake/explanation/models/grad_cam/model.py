# TODO: guided grad cam
# TODO: weakly supervised object localization
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.layer_name = ""
        self._hook = None
        self.activation_map = None

    def set_activation_layer(self, layer_name: str):
        if self._hook is not None:
            self._hook.remove()

        self.layer_name = layer_name
        layer = self.model.get_submodule(layer_name)
        self._hook = layer.register_forward_hook(
            lambda _, __, output: setattr(self, "activation_map", output)
        )

    def forward(
        self,
        x: torch.Tensor,
        target: None | torch.Tensor = None,
        counterfactual: bool = False,
    ) -> torch.Tensor:
        if target is not None:
            assert target.shape == (x.size(0),)

        assert self._hook is not None, "call set_activation_layer first"

        try:
            with torch.enable_grad():
                # B, O
                y = self.model(x)
                # B, h, H, W
                z = self.activation_map
                if target is None:
                    target = y.argmax(1)
                # sum for grad
                y = y.gather(1, target[:, None]).sum()
                # B, h, H, W
                (g,) = torch.autograd.grad(y, z)
                # B, h
                a = g.mean((2, 3))

                if counterfactual:
                    a = -a

                # B, 1, H, W
                y = F.relu(torch.einsum("bdhw,bd->bhw", z, a).unsqueeze(1))

                y = F.interpolate(
                    y,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            return y.detach()
        finally:
            self.activation_map = None
