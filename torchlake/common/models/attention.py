from torch import nn
import torch


class Attention(nn.Module):

    def __init__(
        self,
        encode_dim: int,
        decoder_hidden_dim: int,
        decoder_bidirectional: bool = False,
    ):
        """attention template

        Args:
            encode_dim (int): dimension of encode feature
            decoder_hidden_dim (int): dimenson of hidden state of rnn decoder
            decoder_bidirectional (bool, optional): is rnn decoder bidirectional. Defaults to False.
        """
        super().__init__()
        self.encode_dim = encode_dim
        self.decode_dim = (
            2 * decoder_hidden_dim if decoder_bidirectional else decoder_hidden_dim
        )
        self.decoder_bidirectional = decoder_bidirectional

        self.build_weights()

    def build_weights(self):
        self.a = nn.Linear(self.encode_dim, self.decode_dim)

    def get_attention_weight(self, os: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """get attention weights by current hidden state and source hidden state

        Args:
            os (torch.Tensor): source hidden state, shape is (B, S, eh)
            ht (torch.Tensor): current hidden state, shape is (B, dbi*dh)

        Returns:
            torch.Tensor: attention weights, shape is (B, S)
        """
        # B, dh x B, S, dh => B, S
        a = torch.einsum("bh,bsh->bs", ht, self.a(os))

        # B, S
        return a.softmax(-1)

    def get_context_vector(self, at: torch.Tensor, os: torch.Tensor) -> torch.Tensor:
        """context_vector is the attention weighted sum

        Args:
            at (torch.Tensor): attention weights, shape is (B, S)
            os (torch.Tensor): source hidden state, shape is (B, S, eh)

        Returns:
            torch.Tensor: context vector, shape is (B, 1, eh)
        """
        # B, S x B, S, eh => B, 1, eh
        return torch.einsum("bs,bsh->bh", at, os).unsqueeze_(1)

    def forward(
        self,
        os: torch.Tensor,
        ht: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """global attention forward

        Args:
            os (torch.Tensor): source hidden state, shape is (B, S, eh)
            ht (torch.Tensor): current hidden state, shape is (D, B, dh)

        Returns:
            tuple[torch.Tensor]: context vector(B, 1, eh), attention weight(B, S)
        """
        # only use first layer of hidden states
        top_layer_idx = 2 if self.decoder_bidirectional else 1
        # B, dbi*dh
        ht = ht[:top_layer_idx, :, :].permute(1, 0, 2).flatten(start_dim=1)

        # fix: beam search has shape topk*batch_size, dbi*dh
        if ht.size(0) != os.size(0):
            os = os.repeat(ht.size(0) // os.size(0), 1, 1)

        # B, S
        attentions = self.get_attention_weight(os, ht)

        # B, 1, eh
        z = self.get_context_vector(attentions, os)

        # B, 1, eh / B, S
        return z, attentions
