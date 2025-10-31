import torch
from torch import nn
from torchlake.common.models.model_base import ModelBase

from .network import Highway, SkipRNN, TemporalAttention


class LSTNet(ModelBase):

    def __init__(
        self,
        hidden_dim_c: int = 100,
        hidden_dim_r: int = 100,
        hidden_dim_skip: int = 5,
        output_size: int = 1,
        kernel: int = 6,
        window_size: int = 24 * 7,
        highway_window_size: int = 24,
        skip_window_size: int = 24,
        attention: bool = False,
        dropout_prob: float = 0.2,
        activation: nn.Module | None = None,
    ):
        self.hidden_dim_c = hidden_dim_c
        self.hidden_dim_r = hidden_dim_r
        self.hidden_dim_skip = hidden_dim_skip
        self.kernel = kernel
        self.window_size = window_size
        self.highway_window_size = highway_window_size
        self.skip_window_size = skip_window_size
        self.attention = attention
        self.dropout_prob = dropout_prob
        self.activation = activation
        super().__init__(1, output_size)

    @property
    def feature_dim(self) -> int:
        hidden_dim = self.hidden_dim_r

        if self.skip_window_size > 0 and not self.attention:
            return hidden_dim + self.skip_window_size * self.hidden_dim_skip

        return hidden_dim

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Sequential(
            nn.Conv2d(
                input_channel,
                self.hidden_dim_c,
                (self.kernel, self.output_size),
            ),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_prob),
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleDict(
            {
                "rnn": nn.GRU(self.hidden_dim_c, self.hidden_dim_r),
                "dropout": nn.Dropout(p=self.dropout_prob),
            }
        )

    def build_neck(self, **kwargs):
        self.neck = None

        if self.attention:
            self.neck = TemporalAttention()
        elif self.skip_window_size > 0:
            self.neck = SkipRNN(
                self.hidden_dim_c,
                self.hidden_dim_skip,
                self.kernel,
                self.window_size,
                self.skip_window_size,
                self.dropout_prob,
            )

    def build_head(self, output_size, **kwargs):
        self.head = nn.ModuleDict(
            {
                "output": nn.Linear(self.feature_dim, output_size),
            }
        )

        if self.highway_window_size > 0:
            self.head["highway"] = Highway(self.highway_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x: B, S, C => B, 1, S, C
        x = x.unsqueeze_(1)

        # B, hc, S-k+1, 1
        c = self.foot(x)
        # B, hc, S-k+1
        c = c.squeeze(-1)

        # S-k+1, B, hc
        r = c.permute(2, 0, 1)
        # D=1, B, hr
        _, r = self.blocks["rnn"](r)
        r = self.blocks["dropout"](r)
        # B, hr
        r = r.squeeze(0)
        # B, hr + skip
        if self.neck is not None:
            y = self.neck(c, r)
        # B, hr
        else:
            y = r
        # B, C
        y = self.head["output"](y)
        if self.highway_window_size > 0:
            y = self.head["highway"](x, y)

        if self.activation is not None:
            y = self.activation(y)

        return y
