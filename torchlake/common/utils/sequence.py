import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from ..schemas import NlpContext


def pack_sequence(
    y: torch.Tensor,
    seq_lengths: torch.Tensor,
    padding_idx: int | None = None,
) -> PackedSequence | torch.Tensor:
    if padding_idx is not None and y.size(1) > 1:
        y = pack_padded_sequence(
            y,
            seq_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

    return y


def unpack_sequence(ot: PackedSequence, max_seq_len: int) -> torch.Tensor:
    if isinstance(ot, PackedSequence):
        ot, _ = pad_packed_sequence(
            ot,
            batch_first=True,
            total_length=max_seq_len,
        )

    return ot


def get_input_sequence(
    shape: tuple[int],
    context: NlpContext = NlpContext(),
) -> torch.Tensor:
    return torch.full(shape, context.bos_idx).to(context.device)
