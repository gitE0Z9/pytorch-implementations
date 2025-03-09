import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.models.attention import Attention


# see also https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BahdanauAttention(Attention):
    """[NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/abs/1409.0473)

    not exactly the same, since origianl design is tightly coupled to GRU layer
    """

    def build_weights(self):
        self.q = nn.Linear(self.decode_dim, self.decode_dim)
        # same as self.a of parent, rename as k for consistency
        self.k = nn.Linear(self.encode_dim, self.decode_dim)
        self.v = nn.Linear(self.decode_dim, 1)

    def get_attention_weight(self, os: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """get attention weights by current hidden state and source hidden state

        Args:
            os (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (B, dh)

        Returns:
            torch.Tensor: attention weights, shape is (B, S)
        """
        # B, 1, dh + B, S, dh => B, S
        e: torch.Tensor = self.v(
            torch.tanh(self.q(ht.unsqueeze(-2)) + self.k(os))
        ).squeeze(-1)

        # B, S
        return e.softmax(-1)


class GlobalAttention(Attention):
    """Luong global attention(general)"""


class LocalAttention(Attention):
    def __init__(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool = False,
        context_size: int = 1,
    ):
        """Luong local attention(local-p)

        Args:
            encode_dim (int): dimension of encode feature
            decoder_hidden_dim (int): dimenson of hidden state of rnn decoder
            decoder_bidirectional (bool, optional): is rnn decoder bidirectional. Defaults to False.
            context_size (int, optional): subsampling window size. Defaults to 1.
        """
        self.context_size = context_size
        self.window_size = context_size * 2 + 1
        super().__init__(encode_dim, decoder_hidden_dim, decoder_bidirectional)

    def build_weights(self):
        super().build_weights()
        self.vp = nn.Linear(self.decode_dim, 1)
        self.position_weight = nn.Linear(self.decode_dim, self.decode_dim)

    def get_predicted_position(
        self,
        ht: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """get predicted position of source hidden state

        Args:
            ht (torch.Tensor): current hidden state, shape is (B, h)
            seq_len (int): sequence length of source hidden state

        Returns:
            tuple[torch.Tensor, torch.Tensor]: selected position of source, shape is (B, 1), selected windows, shape is (B, window_size)
        """
        # B, h
        positions: torch.Tensor = self.position_weight(ht).tanh()
        # B, 1
        positions = self.vp(positions).sigmoid() * seq_len

        # B, window_size
        # window_size(2D+1), D is context_size
        # expand positions by window
        windows = (
            positions
            + torch.linspace(
                -self.context_size, self.context_size, self.window_size
            ).to(positions.device)
        ).int()

        # B, 1 # B, window size
        return positions, windows

    def subsample_source_hidden_state(
        self,
        os: torch.Tensor,
        windows: torch.Tensor,
    ) -> torch.Tensor:
        """subsampling source hidden states

        Args:
            os (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            windows (torch.Tensor): windows indices, shape is (B, window size)

        Returns:
            torch.Tensor: windowed source hidden state, shape is (B, W, ebi*eh)
        """
        # B, 1
        batch_indices = torch.arange(os.size(0), device=os.device).unsqueeze_(-1)
        # B, W, ebi*eh
        return F.pad(os, (0, 0, self.context_size, self.context_size))[
            batch_indices, windows, :
        ]

    def get_kernel(
        self,
        positions: torch.Tensor,
        windows: torch.Tensor,
    ) -> torch.Tensor:
        """kernel function to weight source hidden state around predicted position

        Args:
            positions (torch.Tensor): predicted positions, shape is (B, 1)
            windows (torch.Tensor): windows indices, shape is (B, window size)

        Returns:
            torch.Tensor: B, window_size
        """
        # B, window_size
        return (-2 * ((windows - positions) / self.context_size) ** 2).exp()

    def forward(
        self,
        os: torch.Tensor,
        ht: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """local attention forward

        Args:
            os (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            tuple[torch.Tensor]: context vector, shape is (B, 1, dh), attention weights, shape is (D, B, window_size)
        """
        # only use first layer of hidden states
        top_layer_idx = 2 if self.decoder_bidirectional else 1
        # B, dbi*dh
        ht = ht[:top_layer_idx, :, :].permute(1, 0, 2).flatten(start_dim=1)

        # fix: beam search has shape topk*batch_size, dbi*dh
        if ht.size(0) != os.size(0):
            os = os.repeat(ht.size(0) // os.size(0), 1, 1)

        _, seq_len, _ = os.shape

        # B, 1 # B, window size
        positions, windows = self.get_predicted_position(ht, seq_len)

        # B, window_size, ebi*eh
        os = self.subsample_source_hidden_state(os, windows)

        # B, window_size
        kernel = self.get_kernel(positions, windows)

        # B, window_size
        # sigmoid x normal
        attentions = self.get_attention_weight(os, ht) * kernel

        # B, window_size x B, window_size, ebi*eh => B, ebi*eh
        c = self.get_context_vector(attentions, os)

        # B, ebi*eh / B, window_size
        return c, attentions
