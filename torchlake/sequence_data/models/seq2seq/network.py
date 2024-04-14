import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
import torch.nn.functional as F
from ..lstm.model import LstmClassifier


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, ht, hs):
        # hs: B, S, bi * h
        # ht: layers, B, h => B, layers, h as input
        c = self.Va(torch.tanh(self.Wa(ht) + self.Ua(hs)))
        c = c.squeeze(2).unsqueeze(1)

        attentions = c.softmax(-1)
        c = torch.bmm(attentions, hs)

        return c, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


class LuongAttention(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        """[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

        Args:
            latent_dim (int): dimension of latent representation
            num_layers (int, optional): number of LSTM layers. Defaults to 1.
            bidirectional (bool, optional): is bidirectional LSTM. Defaults to False.
        """
        super(LuongAttention, self).__init__()
        self.factor = 2 if bidirectional else 1
        self.layers = self.factor * num_layers

        self.attention_weight = nn.Linear(latent_dim, latent_dim)
        self.context_weight = nn.Linear(2 * latent_dim, latent_dim)

    def split_source_hidden_state(self, hs: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            hs (torch.Tensor): batch_size, seq_len, hidden_dim

        Returns:
            torch.Tensor: layers, B, S, h
        """
        batch_size, seq_len, hidden_dim = hs.shape
        # layers, B, S, h
        return hs.reshape(
            batch_size, seq_len, self.layers, hidden_dim // self.layers
        ).permute(2, 0, 1, 3)

    def get_attention_weight(self, hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            hs (torch.Tensor): _description_
            ht (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # layers, B, S, h
        c = self.attention_weight(hs)

        # layers, B, 1, h x layers, B, h, s => layers, B, 1, S
        c = torch.matmul(ht.unsqueeze(2), c.transpose(-1, -2))
        return c.softmax(-1)

    def output_hidden_state(
        self,
        at: torch.Tensor,
        ht: torch.Tensor,
        hs: torch.Tensor,
    ) -> torch.Tensor:
        """ct = at @ hs, tanh(Wc[ct, ht])

        Args:
            at (torch.Tensor): attention weights,
            ht (torch.Tensor): target hidden state
            hs (torch.Tensor): source hidden state

        Returns:
            torch.Tensor: context vector
        """
        # layers, B, 1, S x layers, B, S, h => layers, B, 1, h
        ct = torch.matmul(at, hs)
        # layers, B, h
        return self.context_weight(torch.cat([ct.squeeze(2), ht], -1)).tanh()


class GlobalAttention(LuongAttention):
    """Luong global attention(general)"""

    def forward(self, hs: torch.Tensor, ht: torch.Tensor) -> tuple[torch.Tensor]:
        """global attention forward

        Args:
            hs (torch.Tensor): B, S, bi * h
            ht (torch.Tensor): layers, B, h

        Returns:
            tuple[torch.Tensor]: attended context vector(layers, B, h), attention weight(layers, B, S)
        """
        # layers, B, S, h
        hs = self.split_source_hidden_state(hs)

        # layers, B, 1, S
        attentions = self.get_attention_weight(hs, ht)

        # layers, B, 1, S x layers, B, S, h => layers, B, 1, h
        c = self.output_hidden_state(attentions, ht, hs)

        # layers, B, h / layers, B, S
        return c, attentions.squeeze(2)


class LocalAttention(LuongAttention):
    """Luong local attention(local-p)"""

    def __init__(self, latent_dim: int, context_size: int = 1, *args, **kwargs):
        """one more argument context_size, other args are the same to global attention

        Args:
            context_size (int, optional): subsampling window size. Defaults to 1.
        """
        super(LocalAttention, self).__init__(latent_dim=latent_dim, *args, **kwargs)
        self.context_size = context_size
        self.window_size = context_size * 2 + 1

        self.v_p = nn.Linear(latent_dim, 1)
        self.position_weight = nn.Linear(latent_dim, latent_dim)

    def get_predicted_position(
        self,
        ht: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            ht (torch.Tensor): _description_
            seq_len (int): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """
        # layers, B, h
        pt = self.position_weight(ht).tanh()
        # layers, B, 1
        pt = self.v_p(pt).sigmoid() * seq_len

        # window_size(2D+1)
        window_index = torch.linspace(
            -self.context_size, self.context_size, self.window_size
        ).to(pt.device)

        # layers, B, window_size
        positions = window_index + pt.int()

        return pt, positions

    def get_kernel(
        self,
        pt: torch.Tensor,
        positions: torch.Tensor,
        hidden_dim: int,
    ) -> torch.Tensor:
        """kernel function to weight source hidden state around predicted position

        Args:
            pt (torch.Tensor): _description_
            positions (torch.Tensor): _description_
            hidden_dim (int): _description_

        Returns:
            torch.Tensor: layers, B, 1, window_size
        """

        # layers, B, 1, window_size
        return (-2 * ((positions - pt) / hidden_dim) ** 2).exp()

    def subsample_source_hidden_state(
        self,
        hs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            hs (torch.Tensor): _description_
            positions (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        batch_size, seq_len, hidden_dim = hs.shape
        dimension_per_layer = hidden_dim // self.layers

        # layers, B, S, h
        hs = hs.reshape(batch_size, seq_len, self.layers, dimension_per_layer).permute(
            2, 0, 1, 3
        )
        # layers, B, window_size, h
        hs = F.pad(hs, (0, 0, self.context_size, self.context_size)).gather(
            2,
            (positions + self.context_size)
            .unsqueeze(-1)
            .repeat(1, 1, 1, dimension_per_layer),
        )

        return hs

    def forward(self, hs: torch.Tensor, ht: torch.Tensor) -> tuple[torch.Tensor]:
        """local attention forward

        Args:
            hs (torch.Tensor): B, S, bi * h
            ht (torch.Tensor): layers, B, h

        Returns:
            tuple[torch.Tensor]: attended context vector
        """
        _, seq_len, hidden_dim = hs.shape

        # layers, B, 1 # layers, B, window_size
        pt, positions = self.get_predicted_position(ht, seq_len)

        # layers, B, 1, window_size
        kernel = self.get_kernel(pt, positions, hidden_dim).unsqueeze(2)

        # layers, B, window_size, h
        hs = self.subsample_source_hidden_state(hs, positions.type(torch.int64))

        # layers, B, 1, window_size
        # sigmoid x normal
        attentions = self.get_attention_weight(hs, ht) * kernel

        # layers, B, 1, window_size x layers, B, window_size, h => layers, B, 1, h
        c = self.output_hidden_state(attentions, ht, hs)

        # layers, B, h / layers, B, window_size
        return c, attentions.squeeze(2)


class Seq2SeqEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
    ):
        """
        將欲翻譯句子轉為隱向量
        """
        super(Seq2SeqEncoder, self).__init__()
        self.rnn = LstmClassifier(
            vocab_size,
            embed_dim,
            hidden_dim,
            output_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            context=context,
        )
        self.rnn.fc = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.rnn.feature_extract(x)


class Seq2SeqAttentionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
    ):
        """
        將欲翻譯句子轉為隱向量
        """
        super(Seq2SeqAttentionEncoder, self).__init__()

        # separate each single layer of lstm to get all t and all layer hidden state
        self.rnns = nn.ModuleList(
            [
                LstmClassifier(
                    vocab_size,
                    embed_dim,
                    hidden_dim,
                    output_size,
                    num_layers=1,
                    bidirectional=bidirectional,
                    context=context,
                )
                for _ in range(num_layers)
            ]
        )

        # remove output layer
        for layer in self.rnns:
            layer.fc = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        features = []
        states_placeholder = []

        states = ()
        for i, layer in enumerate(self.rnns):
            ot, states = layer.feature_extract(x, *states)
            features.append(ot)

            is_multiple_state = isinstance(states, tuple)
            if not is_multiple_state:
                states_placeholder.append(states)
            # for RNN variant has more than one hidden state
            else:
                # to init
                if i == 0:
                    for _ in range(len(states)):
                        states_placeholder.append([])
                for placeholder, state in zip(states_placeholder, states):
                    placeholder.append(state)

        # collect hidden state
        if not is_multiple_state:
            states_placeholder = torch.cat(states_placeholder, 0)
        else:
            for i, placeholder in enumerate(states_placeholder):
                states_placeholder[i] = torch.cat(placeholder, 0)
            states_placeholder = tuple(states_placeholder)

        return torch.cat(features, -1), states_placeholder


class Seq2SeqDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
    ):
        """
        將隱向量與目標句子轉為欲翻譯句子
        """
        super(Seq2SeqDecoder, self).__init__()
        self.rnn = LstmClassifier(
            vocab_size,
            embed_dim,
            hidden_dim,
            output_size,
            num_layers,
            bidirectional,
            context,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return self.rnn(x, h, c, output_state=True)
