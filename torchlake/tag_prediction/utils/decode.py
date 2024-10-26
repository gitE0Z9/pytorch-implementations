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
    x = x.log_softmax(-1)

    # 1, O, O
    transition_score = transition.unsqueeze(0).log_softmax(-1)

    # forward
    backpointers = []

    # P(Y0)
    # B, O, 1
    alpha = torch.rand((batch_size, num_class, 1)).log().to(x.device)
    # start from bos
    alpha[:, context.bos_idx, :] = 0
    alpha[:, context.eos_idx, :] = -1e4
    alpha[:, context.padding_idx, :] = -1e4
    alpha = alpha.log_softmax(1)

    # alpha = P(Y1, Y0 | x) = P(Y1|x) * P(Y0)
    alpha += x[:, 0, :, None]

    for t in range(1, seq_len - 1):
        # batch_size, next_tag, current_tag -> B, O, 1 + 1, O, O
        # P(Y2|Y1) * P(Y1, Y0 | x)
        posterior_t = alpha + transition_score

        # find most likely next tag
        # B, O
        # P(Y2, Y1=y1, Y0 | x)
        posterior_t, bkptr_t = posterior_t.max(dim=-1)
        # debug
        # print(posterior_t.softmax(-1), bkptr_t)
        # P(Y2|x) * P(Y2, Y1=y1, Y0 | x)
        # B, O
        posterior_t += x[:, t, :]
        # S-2 x (B, O)
        backpointers.append(bkptr_t)

        # mask padding token
        if mask is not None:
            # B, 1, 1
            mask_t = mask[:, t : t + 1, None].int()
        else:
            mask_t = torch.zeros((batch_size, 1, 1)).to(x.device)

        # B, O, 1
        alpha = posterior_t.unsqueeze(-1) * (1 - mask_t) + alpha * mask_t

    # to eos ??
    # B, O, 1 + 1, O, 1 => B, O, 1
    # alpha += transition_score[:, :, context.eos_idx, None]

    # get best path and score w.r.t `to` label
    # B, 1
    best_score, best_path = alpha.max(dim=1)
    # debug
    # print(alpha.softmax(1))
    best_score.squeeze_(-1)

    # backward
    # B, S-2, O
    backpointers = torch.stack(backpointers, 1)
    # 1 x (B,1)
    best_path = [best_path]

    for t in range(backpointers.size(1) - 2, -1, -1):
        # B, O
        bkptr_t = backpointers[:, t, :]
        best_path.append(bkptr_t.gather(-1, best_path[-1]))

    # S-2 x (B, 1)
    best_path.reverse()
    best_path.insert(0, torch.full((batch_size, 1), context.bos_idx).to(x.device))
    best_path.append(torch.full((batch_size, 1), context.eos_idx).to(x.device))
    # B, S-2
    best_path = torch.cat(best_path, -1)

    # B,S, # B
    return best_path, best_score
