import torch


def subsampling_probability(word_freqs: torch.Tensor, t: float) -> torch.Tensor:
    """1310.4546 p.4
    subsampling probability in word2vec

    Args:
        word_freqs (torch.Tensor): word frequency
        t (float): subsampling threshold

    Returns:
        torch.Tensor: subsampling probability
    """
    return 1 - (t / word_freqs).sqrt()


def subsampling(
    x: torch.Tensor,
    subsampling_probs: torch.Tensor,
    unk_idx: int,
) -> torch.Tensor:
    """1310.4546 p.4
    subsampling tokens with the probability of word frequency formula

    Args:
        x (torch.Tensor): token indices, shape is (batch_size, neighbor_size, #subsequence)
        subsampling_probs (torch.Tensor): probability distribution of each token with formula in paper
        unk_idx (int): unknown index as masked index

    Returns:
        torch.Tensor: masked input, shape is (batch_size, neighbor_size, #subsequence)
    """
    return x.masked_fill(subsampling_probs[x].bernoulli().bool(), unk_idx)
