import torch
from torchlake.common.models.attention import Attention


class HardAttention(Attention):

    def get_context_vector(self, at: torch.Tensor, os: torch.Tensor) -> torch.Tensor:
        """context_vector is an attention-weights-sampled visual patch

        Args:
            at (torch.Tensor): attention weights, shape is (B, S)
            os (torch.Tensor): source hidden state, shape is (B, S, eh)

        Returns:
            torch.Tensor: context vector, shape is (B, 1, eh)
        """
        batch_size, _ = at.shape
        # B, 1
        s = at.multinomial(1)
        # B, 1, eh
        return os[torch.arange(batch_size, device=os.device).unsqueeze_(-1), s, :]


class SoftAttention(Attention):
    """Deterministic soft attention"""
