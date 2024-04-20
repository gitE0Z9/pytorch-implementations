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
