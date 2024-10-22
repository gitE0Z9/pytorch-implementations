import torch
from torchlake.common.schemas.nlp import NlpContext


def viterbi_decode(
    x: torch.Tensor,
    transition: torch.Tensor,
    mask: torch.Tensor | None = None,
    context: NlpContext = NlpContext(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """forward backward algorithm

    viterbi forward: find best path
    backward: retrieve token

    Args:
        x (torch.Tensor): predicted probability, shape is (batch_size, sequence_length, output_size)
        transition (torch.Tensor): transition matrix, shape is (output_size, output_size)
        mask (torch.Tensor | None, optional): mask for padding index, shape is (batch_size, sequence_length). Defaults to None.
        context (NlpContext, optional): nlp context. Defaults to NlpContext().

    Returns:
        tuple[torch.Tensor, torch.Tensor]: score of path, path
    """
    batch_size, seq_len, num_class = x.shape

    backpointers = []

    # 1, O, O
    transition_score = transition.unsqueeze(0).log_softmax(-1)

    # forward

    # P(y_t, ...., y_0|x) = PI{t:0..T} P(y_t|y_t-1, x) * P(y_t-1)
    log_likelihood = torch.full((batch_size, num_class, 1), -1e4).to(x.device)
    # start from bos
    log_likelihood[:, context.bos_idx, :] = 0

    for t in range(seq_len):
        # batch_size, next_tag, current_tag -> B, O, 1 + 1, O, O
        posterior_t = log_likelihood + transition_score

        # find most likely next tag
        # B, O
        posterior_t, bkptr_t = posterior_t.max(dim=-1)

        # P(y_t-1|y_t) = P(y_t) * P(y_t|y_t-1) * P(y_t-1)
        # B, O
        posterior_t += x[:, t, :]
        # S x (B, O)
        backpointers.append(bkptr_t)

        # mask padding token
        if mask is not None:
            # B, 1
            mask_t = mask[:, t : t + 1].int()
        else:
            mask_t = 0

        # B, O, 1
        log_likelihood = (posterior_t * (1 - mask_t)).unsqueeze(-1)

    # to eos ??
    # B, O, 1 + 1, O, 1 => B, O, 1
    log_likelihood += transition_score[:, :, context.eos_idx].unsqueeze(-1)

    # get best path and score w.r.t `to` label
    # B, 1
    best_score, best_path = log_likelihood.max(dim=1)

    # backward
    # B, S, O
    backpointers = torch.stack(backpointers, 1)
    # B x 1
    best_path: list[list[int]] = best_path.tolist()
    # 1
    for batch_idx, node in enumerate(best_path):
        path = [node]
        # seq_len_i = seq_len - mask[batch_idx].sum() if mask is not None else seq_len

        # reverse order to backward for retrieving token
        # O
        # for ptr_t in reversed(backpointers[batch_idx, :seq_len_i]):
        for ptr_t in reversed(backpointers[batch_idx]):
            # 1
            path.append(ptr_t[path[-1]].item())
        # pop first tag
        # best_path[batch_idx].pop()  # B x (S+1) -> B x (S)
        # reverse order back to forward
        # best_path[batch_idx].reverse()
        best_path[batch_idx] = path[1:][::-1]

    # B,S, # B
    return torch.Tensor(best_path), best_score.squeeze_(-1)
