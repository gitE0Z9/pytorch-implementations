import torch
import torch.nn.functional as F
from torch import nn


def normalize(w: torch.Tensor) -> torch.Tensor:
    return F.normalize(w.view(w.size(0), -1), p=2, dim=-1).view_as(w)


class DiracConv2d(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
    ):
        super().__init__()
        self.kernel = kernel
        self.w = nn.Parameter(torch.rand(output_channel, input_channel, kernel, kernel))
        self.b = nn.Parameter(torch.rand(output_channel))

        self.norm = normalize
        self.act = nn.ReLU(True)

        # different from my understanding
        self.identity_scale = nn.Parameter(torch.ones((output_channel, 1, 1, 1)))
        self.norm_scale = nn.Parameter(torch.full((output_channel, 1, 1, 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(
            x,
            weight=self.identity_scale * self.w + self.norm_scale * self.norm(self.w),
            bias=self.b,
            padding=self.w.size(2) // 2,
        )

        return self.act(y)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        for key, parameter in self._parameters.items():
            param_str = repr(parameter.shape)
            param_str = nn.modules.module._addindent(param_str, 2)
            child_lines.append("(" + key + "): " + param_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
