from operator import itemgetter

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import HuffmanNode, build_huffman_tree


class NegativeSampling(nn.Module):
    def __init__(
        self,
        word_freqs: torch.Tensor,
        embed_dim: int,
        vocab_size: int,
        negative_ratio: int = 5,
        power: float = 0.75,
        replacement: bool = True,
        context: NLPContext | None = None,
    ):
        """negative sampling loss

        Args:
            word_freqs (torch.Tensor): word frequency
            embed_dim (int): embedding dimension
            vocab_size (int): vocabulary size
            negative_ratio (int, optional): negative sample size compare to positive sample size. Defaults to 5.
            power (float, optional): power parameter. Defaults to 0.75.
            replacement (bool, optional): enable replacement for faster sampling. Defaluts to True.
            context (NLPContext, optional): NLP context. Defaults to None.
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
        self.fc = nn.Parameter(torch.rand((vocab_size, embed_dim)))

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
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, subseq)

        Returns:
            torch.Tensor: sampled token by noise distribution, shape is (batch_size, 1 or neighbor_size, subseq, #neg)
        """
        n: int = target.numel()

        if self.replacement:
            # (n, #neg)
            y = self.distribution.multinomial(
                n * self.negative_ratio, replacement=True
            ).view(n, self.negative_ratio)

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

    def forward(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """compute negative sampling loss

        Args:
            embedding (torch.Tensor): shape(batch_size, 1 or neighbor_size, subseq, embed_dim)
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, subseq)

        Returns:
            torch.Tensor: float
        """
        # (B, neighbor_size, subseq, h) x (B, neighbor_size, subseq, h)
        # => (B, neighbor_size, subseq)
        positive_logits = torch.einsum(
            "bcsh, bcsh -> bcs",
            embedding,
            self.fc[target],
        )

        # (B, neighbor_size, subseq, #negative)
        negative_sample = self.sample(target)
        # (B, neighbor_size, subseq, 1, h) x (B, neighbor_size, subseq, h, #negative)
        # => (B, neighbor_size, subseq, 1, #negative)
        negative_logits = torch.einsum(
            "bcsh, bcsnh -> bcsn",
            embedding,
            self.fc[negative_sample],
        )

        positive_loss = binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
            reduction="sum",
        )

        negative_loss = binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits),
            reduction="sum",
        )

        return positive_loss + negative_loss


class HierarchicalSoftmax(nn.Module):
    def __init__(
        self,
        word_counts: torch.Tensor,
        embed_dim: int,
        vocab_size: int,
        context: NLPContext | None = None,
    ):
        """hierarchical softmax loss

        Args:
            word_counts (torch.Tensor): word counts
            embed_dim (int): embedding dimension
            vocab_size (int): vocabulary size
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        super().__init__()
        self.vocab_size = vocab_size
        self.context = context
        # leaf size + combined node size = vocab + (vocab - 1)
        # self.tree_size = 2 * vocab_size - 1

        tree = self.build_tree(word_counts)
        self.path_indices, self.path_codes = self.get_paths(tree)
        # vectors of path nodes
        self.fc = nn.Parameter(torch.rand((vocab_size - 1, embed_dim)))

    def build_tree(self, word_counts: torch.Tensor) -> HuffmanNode:
        """build huffman tree by word counts

        Returns:
            HuffmanNode: root of huffman tree
        """
        root = build_huffman_tree(word_counts.tolist())

        # combined node size is (vocab - 1)
        # so 0-index is (vocab - 2)
        assert root.value == (
            2 * self.vocab_size - 2
        ), "Size of intermediate nodes is incorrect."

        return root

    def get_paths(
        self,
        root: HuffmanNode,
    ) -> tuple[dict[int, torch.LongTensor], dict[int, torch.Tensor]]:
        """get paths to leaves of huffman tree

        Args:
            root (HuffmanNode): root of huffman tree

        Returns:
            tuple[dict[int, torch.LongTensor], dict[int, torch.Tensor]]: intermediate indices to leaves and codes
        """
        assert root is not None, "root should be a HuffmanNode"
        N = self.vocab_size

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

    def forward(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            embedding (torch.Tensor): embedding vectors, shape(batch_size, 1 or neighbor_size, #subsequence, embed_dim)
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence)

        Returns:
            torch.Tensor: loss
        """
        device = self.context.device
        # (N = B * c * #subseq), h
        embedding = embedding.view(-1, embedding.size(-1))

        # target mapping and concat
        # N
        y = target.view(-1).tolist()
        path_indices = itemgetter(*y)(self.path_indices)
        path_codes = itemgetter(*y)(self.path_codes)

        # indices, shape is ?
        internal_indices = torch.cat(path_indices).to(device)

        # index of sample
        # for example return 1, 1, 2, 3
        # if sample 1 with tree depth 2, sample 2 and 3 with tree depth 1
        sample_indices = torch.repeat_interleave(
            torch.arange(len(path_indices)),
            torch.LongTensor([path_idx.shape[0] for path_idx in path_indices]),
        ).to(device)

        # ?
        pred = torch.einsum(
            "xh, xh -> x",
            embedding[sample_indices],
            self.fc[internal_indices],
        )

        # ?
        target = torch.cat(path_codes).to(device)

        return binary_cross_entropy_with_logits(pred, target)
