import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class DiscretizedLogisticMixture(nn.Module):
    def __init__(
        self,
        input_channel: int,
        scale: float = 1 / 255,
        epsilon: float = 1e-3,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ):
        """discretized logistic mixture distribution

        Args:
            input_channel (int): input channel
            reduction (Literal["none", "sum", "mean"], optional): reduction. Default is "mean".
        """
        super().__init__()
        self.input_channel = input_channel
        self.scale = scale
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(
        self,
        yhat: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """forward

        Args:
            yhat (torch.Tensor): shape is (B, K, (1 + 2C + fib(C-1)*C), H, W).
            x (torch.Tensor): shape is (B, C, H, W).

        Returns:
            torch.Tensor: mixture prob, shape is (B, C, H, W)
        """
        ### decode

        C = self.input_channel
        # B, K, 1, H, W
        pi: torch.Tensor = yhat[:, :, 0:1, :, :]
        mu, log_sigma, coef = (
            # B, K, C, H, W
            yhat[:, :, 1 : 1 + C],
            # B, K, C, H, W
            yhat[:, :, 1 + C : 1 + 2 * C],
            # tanh is not in the paper
            # B, K, sum(C-1 + ... + 1), H, W
            yhat[:, :, 1 + 2 * C :].tanh(),
        )
        # not in the paper, important !!!
        # https://github.com/openai/pixel-cnn/blob/bbc15688dd37934a12c2759cf2b34975e15901d9/pixel_cnn_pp/nn.py#L54
        # sigma can down to 0.0009
        log_sigma = log_sigma.clamp(min=-7)
        precision = (-log_sigma).exp()

        ### prob

        # B, C, H, W => B, 1, C, H, W
        x = x[:, None]
        # B, K, 1, H, W => B, K, C, H, W
        mu_cond = mu[:, :, 0:1, :, :]
        cursor = 0
        for i in range(1, C):
            mu_cond = torch.cat(
                [
                    mu_cond,
                    mu[:, :, i : i + 1]
                    + (coef[:, :, cursor : cursor + i] * x[:, :, :i]).sum(
                        2, keepdim=True
                    ),
                ],
                2,
            )
            cursor += i

        # B, K, C, H, W
        x_tilde = x - mu_cond
        rightside = precision * (x_tilde + (self.scale / 2))
        leftside = precision * (x_tilde - (self.scale / 2))

        # softplus is not in the paper, important !!!
        # softplus(x) = ln(1+exp(x)) = -ln(1-σ(x))
        # ln(σ(x)) = x - softplus(x)

        # for x -> 1
        # 1 - F.sigmoid(leftside)
        log_cdf_rightside = -F.softplus(leftside)

        # for x -> 0
        # F.sigmoid(rightside)
        log_cdf_leftside = rightside - F.softplus(rightside)

        # integral approximation is not in the paper, important !!!
        # for 0 < x < 1
        # torch.log(F.sigmoid(rightside) - F.sigmoid(leftside))
        # since bin is small, cdf between rightside and leftside is replaced with integral approximation with logistic_pdf(midpoint)
        log_pdf_mid = (
            x_tilde * precision
            - 2 * F.softplus(x_tilde * precision)
            + math.log(self.scale)
        )

        # B, K, C, H, W
        log_probs = torch.where(
            x > (1 - self.epsilon),
            log_cdf_rightside,
            torch.where(x < self.epsilon, log_cdf_leftside, log_pdf_mid),
        )

        # (B, K, C, H, W) + (B, K, 1, H, W) => (B, C, H, W)
        # different from the original code, I add pi before sum instead of sum then add pi
        nll = -(log_probs + F.log_softmax(pi, dim=1)).logsumexp(1)

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "none":
            return nll
        else:
            raise NotImplementedError
