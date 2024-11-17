from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from ..base.wrapper import SequenceModelFullFeatureExtractor
from ..base.rnn_discriminator import RNNDiscriminator


# see also https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BahdanauAttention(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool = False,
    ):
        """[NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/abs/1409.0473)
        In original design, the encoder is bidirectional and the decoder is unidirectional
        so we extend it a bit and deal with asymmetric encoder/decoder

        not exactly the same, since origianl design is tightly coupled to GRU layer

        Args:
            encoder_hidden_dim (int): encoder hidden dimension
            decoder_hidden_dim (int): decoder hidden dimension
            encoder_bidirectional (bool, optional): is encoder bidirectional. Defaults to False.
        """
        super().__init__()
        self.encoder_factor = 2 if encoder_bidirectional else 1

        self.q = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.k = nn.Linear(self.encoder_factor * encoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1)
        self.c = nn.Linear(self.encoder_factor * encoder_hidden_dim, decoder_hidden_dim)

    def get_attention_weight(self, hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """get attention weights by current hidden state and source hidden state

        Args:
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            torch.Tensor: attention weights, shape is (D, B, S)
        """
        # D, B, 1, dh + B, S, dh => D, B, S
        e: torch.Tensor = self.v(
            torch.tanh(self.q(ht.unsqueeze(-2)) + self.k(hs))
        ).squeeze(-1)

        # D, B, S
        return e.softmax(-1)

    def output_hidden_state(self, at: torch.Tensor, hs: torch.Tensor) -> torch.Tensor:
        # D, B, S x B, S, ebi*eh => D, B, ebi*eh
        c = torch.einsum("dbs,bsg->dbg", at, hs)

        # D, B, dh
        return self.c(c)

    def forward(
        self,
        hs: torch.Tensor,
        ht: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward

        Args:
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: context vector, shape is (D, B, dh). attention, shape is (layers, dbi, B, S)
        """
        # D, B, S
        attentions = self.get_attention_weight(hs, ht)

        # D, B, dh
        c = self.output_hidden_state(attentions, hs)

        # D, B, dh / D, B, S
        return c, attentions


class LuongAttention(nn.Module):

    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool = False,
    ):
        """[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

        Args:
            encoder_hidden_dim (int): encoder hidden dimension
            decoder_hidden_dim (int): decoder hidden dimension
            encoder_bidirectional (bool, optional): is encoder bidirectional. Defaults to False.
        """
        super().__init__()
        self.encoder_factor = 2 if encoder_bidirectional else 1

        self.k = nn.Linear(
            self.encoder_factor * encoder_hidden_dim,
            decoder_hidden_dim,
        )
        self.c = nn.Linear(
            self.encoder_factor * encoder_hidden_dim + decoder_hidden_dim,
            decoder_hidden_dim,
        )

    def get_attention_weight(self, hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """get attention weights by current hidden state and source hidden state

        Args:
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            torch.Tensor: attention weights, shape is (D, B, S)
        """
        # this is general form, I left dot, concat, location form for you
        # D, B, dh x B, S, dh => D, B, S
        a = torch.einsum("dbh,bsh->dbs", ht, self.k(hs))

        # D, B, S
        return a.softmax(-1)

    def output_hidden_state(
        self,
        at: torch.Tensor,
        hs: torch.Tensor,
        ht: torch.Tensor,
    ) -> torch.Tensor:
        """ct = at @ hs, tanh(Wc[ct || ht])

        Args:
            at (torch.Tensor): attention weights, shape is (D, B, S)
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            torch.Tensor: context vector, shape is (D, B, dh)
        """
        # D, B, S x B, S, ebi*eh => D, B, ebi*eh
        ct = torch.einsum("dbs,bsh->dbh", at, hs)

        # D, B, (ebi*eh + dh) => D, B, dh
        return self.c(torch.cat([ct, ht], -1)).tanh()


class GlobalAttention(LuongAttention):
    """Luong global attention(general)"""

    def forward(self, hs: torch.Tensor, ht: torch.Tensor) -> tuple[torch.Tensor]:
        """global attention forward

        Args:
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            tuple[torch.Tensor]: context vector(D, B, dh), attention weight(D, B, S)
        """
        # D, B, S
        attentions = self.get_attention_weight(hs, ht)

        # D, B, dh
        c = self.output_hidden_state(attentions, hs, ht)

        # D, B, dh / D, B, S
        return c, attentions


class LocalAttention(LuongAttention):
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_bidirectional: bool = False,
        context_size: int = 1,
    ):
        """Luong local attention(local-p)

        Args:
            encoder_hidden_dim (int): encoder hidden dimension
            decoder_hidden_dim (int): decoder hidden dimension
            encoder_bidirectional (bool, optional): is encoder bidirectional. Defaults to False.
            context_size (int, optional): subsampling window size. Defaults to 1.
        """
        self.context_size = context_size
        self.window_size = context_size * 2 + 1
        super().__init__(encoder_hidden_dim, decoder_hidden_dim, encoder_bidirectional)

        self.vp = nn.Linear(decoder_hidden_dim, 1)
        self.position_weight = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)

    def get_predicted_position(
        self,
        ht: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """get predicted position of source hidden state

        Args:
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)
            seq_len (int): sequence length of source hidden state

        Returns:
            tuple[torch.Tensor, torch.Tensor]: selected position of source, shape is (D, B, 1), selected windows, shape is (D, B, window_size)
        """
        # D, B, h
        positions: torch.Tensor = self.position_weight(ht).tanh()
        # D, B, 1
        positions = self.vp(positions).sigmoid() * seq_len

        # window_size(2D+1)
        window_index = torch.linspace(
            -self.context_size, self.context_size, self.window_size
        ).to(positions.device)

        # D, B, window_size
        windows = window_index + positions

        # D, B, 1 # D, B, window size
        return positions, windows.int()

    def get_kernel(
        self,
        positions: torch.Tensor,
        windows: torch.Tensor,
    ) -> torch.Tensor:
        """kernel function to weight source hidden state around predicted position

        Args:
            positions (torch.Tensor): predicted positions, shape is (D, B, 1)
            windows (torch.Tensor): windows indices, shape is (D, B, window size)

        Returns:
            torch.Tensor: D, B, window_size
        """

        # D, B, window_size
        return (-2 * ((windows - positions) / self.context_size) ** 2).exp()

    def subsample_source_hidden_state(
        self,
        hs: torch.Tensor,
        windows: torch.Tensor,
    ) -> torch.Tensor:
        """subsampling source hidden states

        Args:
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            windows (torch.Tensor): windows indices, shape is (D, B, window size)

        Returns:
            torch.Tensor: windowed source hidden state, shape is (D, B, W, ebi*eh)
        """
        batch_indices = torch.arange(hs.size(0), device=hs.device).view(1, -1, 1)
        return F.pad(hs, (0, 0, self.context_size, self.context_size))[
            batch_indices, windows, :
        ]

    def get_attention_weight(self, hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """get attention weights by current hidden state and source hidden state

        Args:
            hs (torch.Tensor): source hidden state, shape is (D, B, window size, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            torch.Tensor: attention weights, shape is (D, B, window size)
        """
        # D, B, dh x D, B, window size, dh => D, B, window size
        a = torch.einsum("dbh,dbwh->dbw", ht, self.k(hs))

        # D, B, window size
        return a.softmax(-1)

    def output_hidden_state(
        self,
        at: torch.Tensor,
        hs: torch.Tensor,
        ht: torch.Tensor,
    ) -> torch.Tensor:
        """ct = at @ hs, tanh(Wc[ct || ht])

        Args:
            at (torch.Tensor): attention weights, shape is (D, B, window_size)
            hs (torch.Tensor): source hidden state, shape is (D, B, window_size, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            torch.Tensor: context vector, shape is (D, B, dh)
        """
        # D, B, W x D, B, W, ebi*eh => D, B, ebi*eh
        ct = torch.einsum("dbw,dbwh->dbh", at, hs)

        # D, B, (ebi*eh + dh) => D, B, dh
        return self.c(torch.cat([ct, ht], -1)).tanh()

    def forward(
        self,
        hs: torch.Tensor,
        ht: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """local attention forward

        Args:
            hs (torch.Tensor): source hidden state, shape is (B, S, ebi*eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            tuple[torch.Tensor]: context vector, shape is (D, B, dh), attention weights, shape is (D, B, window_size)
        """
        _, seq_len, _ = hs.shape

        # D, B, 1 # D, B, window size
        positions, windows = self.get_predicted_position(ht, seq_len)

        # D, B, window_size, ebi*eh
        hs = self.subsample_source_hidden_state(hs, windows)

        # D, B, window_size
        kernel = self.get_kernel(positions, windows)

        # D, B, window_size
        # sigmoid x normal
        attentions = self.get_attention_weight(hs, ht) * kernel

        # D, B, window_size x D, B, window_size, ebi*eh => D, B, dh
        c = self.output_hidden_state(attentions, hs, ht)

        # D, B, dh / D, B, window_size
        return c, attentions
