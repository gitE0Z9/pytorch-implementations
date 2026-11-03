from operator import itemgetter
from typing import Literal

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import HuffmanNode, build_huffman_tree


class NegativeSampling(nn.Module):
    def __init__(
        self,
        word_freqs: torch.Tensor,
        negative_ratio: int = 5,
        power: float = 0.75,
        replacement: bool = True,
        context: NLPContext | None = None,
        reduction: Literal["sum", "mean"] | None = "sum",
    ):
        """negative sampling loss

        Args:
            word_freqs (torch.Tensor): word frequency
            negative_ratio (int, optional): negative sample size compare to positive sample size. Defaults to 5.
            power (float, optional): power parameter. Defaults to 0.75.
            replacement (bool, optional): enable replacement for faster sampling. Defaluts to True.
            context (NLPContext, optional): NLP context. Defaults to None.
            reduction (Literal["sum", "mean"] | None, optional): reduction. Defaults to "sum".
        """
        assert negative_ratio > 0, "negative ratio should be higher than 0"

        if context is None:
            context = NLPContext()

        super().__init__()
        self.context = context
        self.negative_ratio = negative_ratio
        self.power = power
        self.word_freqs = word_freqs
        self.distribution = self.get_distribution().to(context.device)
        self.vocab_size = self.distribution.numel()
        self.replacement = replacement
        self.reduction = reduction

    def get_distribution(self) -> torch.Tensor:
        """1310.4546 p.4
        noise distribution of word frequency formula

        Returns:
            torch.Tensor: noise distribution, shape is (vocab_size)
        """
        return nn.functional.normalize(self.word_freqs.pow(self.power), p=1, dim=0)

    def sample(self, target: torch.Tensor) -> torch.Tensor:
        """negative sampling by noise distribution

        Args:
            target (torch.Tensor): shape(batch_size*subseq, 1 or neighbor_size)

        Returns:
            torch.Tensor: sampled token by noise distribution, shape is (batch_size*subseq, 1 or neighbor_size, #neg)
        """
        n: int = target.numel()

        if self.replacement:
            # TODO: skipgram use target view as well
            # cbow could benefit from view but not skipgram
            # (n, #neg)
            y = self.distribution.multinomial(
                n * self.negative_ratio, replacement=True
            ).reshape(n, self.negative_ratio)

            collision = y == target.view(-1, 1)
            while collision.any():
                y[collision] = self.distribution.multinomial(
                    collision.sum().item(), replacement=True
                )
                collision = y == target.view(-1, 1)

            # (B, neighbor_size, subseq, #neg)
            return y.view(*target.shape, self.negative_ratio)
        else:
            y = self.distribution.repeat(n, 1)
            # remove positive vocab
            # TODO: skipgram use target view as well
            # cbow could benefit from view but not skipgram
            y[torch.arange(n), target.reshape(-1)] = 0

            return (
                y
                # only 2 dim supported
                .multinomial(self.negative_ratio)
                # (B, context-1, subseq, neg)
                .view(*target.shape, self.negative_ratio)
            )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        output_embedding: nn.Embedding,
    ) -> torch.Tensor:
        """compute negative sampling loss

        Args:
            pred (torch.Tensor): shape(batch_size*subseq, 1 or neighbor_size, embed_dim)
            gt (torch.Tensor): shape(batch_size*subseq, 1 or neighbor_size)
            output_embedding (nn.Embedding): embedding layer for output matrix

        Returns:
            torch.Tensor: float
        """
        # N = B*subseq / sample size
        # W = 1 or neighbor_size / window size

        # (N, W, h) x (N, W, h) => (N, W)
        positive_logits = torch.einsum(
            "nwh, nwh -> nw",
            pred,
            output_embedding(gt),
        )

        # (N, W, #negative)
        negative_sample = self.sample(gt)
        # (N, W, h) x (N, W, #negative, h) => (N, W, #negative)
        negative_logits = torch.einsum(
            "nwh, nwrh -> nwr",
            pred,
            output_embedding(negative_sample),
        )

        positive_loss = binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
            reduction=self.reduction if self.reduction is not None else "none",
        )

        negative_loss = binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits),
            reduction=self.reduction if self.reduction is not None else "none",
        )

        if self.reduction is None:
            return positive_loss, negative_loss
        else:
            return positive_loss + negative_loss


class HierarchicalSoftmaxLoss(nn.Module):
    def __init__(
        self,
        root: HuffmanNode,
        vocab_size: int,
        context: NLPContext | None = None,
    ):
        """hierarchical softmax loss

        Args:
            root (HuffmanNode): the root of a Huffman tree based on word counts
            vocab_size (int): vocabulary size
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        super().__init__()
        self.context = context

        self.path_indices, self.path_codes = self.get_paths(root, vocab_size)

    @staticmethod
    def get_paths(
        root: HuffmanNode,
        vocab_size: int,
    ) -> tuple[dict[int, torch.LongTensor], dict[int, torch.Tensor]]:
        """get paths to leaves of huffman tree

        Args:
            root (HuffmanNode): root of huffman tree
            vocab_size (int): vocabulary size

        Returns:
            tuple[dict[int, torch.LongTensor], dict[int, torch.Tensor]]: intermediate indices to leaves and codes
        """
        assert root is not None, "root should be a HuffmanNode"
        N = vocab_size

        # shape is depth
        path_indices = {}
        # shape is depth; value is 0 or 1
        path_codes = {}

        def traverse(node: HuffmanNode, indices: list[int], codes: list[int]):
            if node.is_leaf:
                leaf_idx = node.value
                path_indices[leaf_idx] = torch.LongTensor(indices)
                path_codes[leaf_idx] = torch.FloatTensor(codes)
                return

            node_idx = node.value - N
            if node.left:
                traverse(node.left, indices + [node_idx], codes + [0])
            if node.right:
                traverse(node.right, indices + [node_idx], codes + [1])

        traverse(root, [], [])

        return path_indices, path_codes

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        output_embedding: nn.Embedding,
    ) -> torch.Tensor:
        """forward

        Args:
            pred (torch.Tensor): embedding vectors, shape(batch_size*#subsequence, 1 or neighbor_size, embed_dim)
            gt (torch.Tensor): shape(batch_size*#subsequence, 1 or neighbor_size)
            output_embedding (nn.Embedding): embedding layer for output matrix

        Returns:
            torch.Tensor: loss
        """
        device = self.context.device
        # (N = B * #subseq * c), h
        pred = pred.view(-1, pred.size(-1))

        # TODO: skipgram use target view as well
        # cbow could benefit from view but not skipgram
        # target mapping and concat
        # N
        gt = gt.reshape(-1).tolist()
        gt_path_indices: list[torch.LongTensor] = itemgetter(*gt)(self.path_indices)
        # indices of prediction, for example
        # if the path length to vocab 1 is 2, the path length to vocab 2 and 3 is 1, return 1, 1, 2, 3
        pred_indices = torch.repeat_interleave(
            torch.arange(len(gt_path_indices), device=device),
            torch.tensor(
                [path_idx.shape[0] for path_idx in gt_path_indices],
                dtype=torch.long,
                device=device,
            ),
        )
        # indices, shape is ?
        gt_path_indices = torch.cat(gt_path_indices).to(device)

        # ?
        pred = torch.einsum(
            "xh,xh->x",
            pred[pred_indices],
            output_embedding(gt_path_indices),
        )

        # ?
        path_codes = itemgetter(*gt)(self.path_codes)
        gt = torch.cat(path_codes).to(device)

        return binary_cross_entropy_with_logits(pred, gt)
