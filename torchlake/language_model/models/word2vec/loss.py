import heapq
from operator import itemgetter

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, one_hot
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.tree import HuffmanNode


class NegativeSampling(nn.Module):
    def __init__(
        self,
        word_freqs: torch.Tensor,
        embed_dim: int,
        vocab_size: int,
        negative_ratio: int = 5,
        power: float = 0.75,
        context: NlpContext = NlpContext(),
    ):
        """negative sampling loss

        Args:
            word_freqs (torch.Tensor): word frequency
            embed_dim (int): embedding dimension
            vocab_size (int): vocabulary size
            negative_ratio (int, optional): negative sample size compare to positive sample size. Defaults to 5.
            power (float, optional): power parameter. Defaults to 0.75.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(NegativeSampling, self).__init__()
        self.context = context
        self.negative_ratio = negative_ratio
        self.power = power
        self.word_freqs = word_freqs
        self.distribution = self.get_distribution().to(context.device)
        self.vocab_size = self.distribution.numel()
        self.fc = nn.Parameter(torch.rand((vocab_size, embed_dim)))

        assert negative_ratio > 0, "negative ratio should be higher than 0"

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
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence)

        Returns:
            torch.Tensor: sampled token by noise distribution, shape is (B, context-1, subseq, #neg)
        """
        # (B, context-1, subseq), #neg
        return (
            self.distribution.repeat(*target.shape, 1)
            .masked_fill(
                one_hot(target, self.vocab_size).bool(), 0
            )  # remove positive vocab
            .view(-1, self.vocab_size)  # only 2 dim supported
            .multinomial(self.negative_ratio)  # last dim for dist
            .view(*target.shape, self.negative_ratio)
        )

    def forward(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """compute negative sampling loss

        Args:
            embedding (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence, embed_dim)
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence)

        Returns:
            torch.Tensor: float
        """
        batch_size = embedding.size(0)

        # B, context-1, subseq, h x B, context-1, subseq, h
        # => B, context-1, subseq
        positive_logits = torch.einsum(
            "bcsh, bcsh -> bcs",
            embedding,
            self.fc[target],
        )

        # B, context-1, subseq, #negative
        negative_sample = self.sample(target)
        # B, context-1, subseq, 1, h x B, context-1, subseq, h, #negative
        # => B, context-1, subseq, 1, #negative
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

        negative_loss = (
            binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
                reduction="sum",
            )
            / batch_size
        )

        return positive_loss + negative_loss


class HierarchicalSoftmax(nn.Module):
    def __init__(
        self,
        word_counts: torch.Tensor,
        embed_dim: int,
        vocab_size: int,
        context: NlpContext = NlpContext(),
    ):
        """hierarchical softmax loss

        Args:
            word_counts (torch.Tensor): word counts
            embed_dim (int): embedding dimension
            vocab_size (int): vocabulary size
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(HierarchicalSoftmax, self).__init__()
        self.context = context
        self.word_counts = word_counts
        self.vocab_size = vocab_size
        # leaf size + combined node size = vocab + (vocab - 1)
        self.tree_size = 2 * vocab_size - 1

        self.tree = self.build_tree()
        self.paths = self.build_huffman_path(self.tree)
        self.fc = nn.Parameter(torch.rand((vocab_size - 1, embed_dim)))

    def build_tree(self) -> HuffmanNode:
        """build huffman tree by word counts

        Returns:
            HuffmanNode: root of huffman tree
        """

        nodes: list[HuffmanNode] = [
            HuffmanNode(index, count)
            for index, count in enumerate(self.word_counts.tolist())
        ]
        heapq.heapify(nodes)

        # index for combined node
        internal_index = 0
        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            combined_node = HuffmanNode(None, left.freq + right.freq, internal_index)
            combined_node.left = left
            combined_node.right = right
            internal_index += 1
            heapq.heappush(nodes, combined_node)

        # combined node size is (vocab - 1), another -1 for being root
        if nodes[0].internal_index != (self.vocab_size - 2):
            print("Size of inner nodes is incorrect.")

        return nodes[0]

    def build_huffman_path(
        self,
        root: HuffmanNode,
        current_codes: list[int] = [],
        current_internal_indices: list[int] = [],
        huffman_codes={},
    ) -> dict[int, torch.Tensor]:
        """build path information to leaves on huffman tree

        Args:
            root (HuffmanNode): root of huffman tree
            current_codes (list[int], optional): list of 1 and 0 until this node. Defaults to [].
            current_internal_indices (list[int], optional): list of internal indices(weight indices) utils this node. Defaults to [].
            huffman_codes (dict, optional): dict of codes and indices of leaves until this code. Defaults to {}.

        Returns:
            dict[int, torch.Tensor]: dict of codes and indices of leaves until this code
        """
        if root is None:
            return huffman_codes

        # reach leaf
        if root.value is not None:
            huffman_codes[root.value] = {
                # shape is depth; value btw 1 or 0
                "code": torch.Tensor(current_codes).to(self.context.device),
                # shape is depth
                "indices": torch.LongTensor(current_internal_indices).to(
                    self.context.device
                ),
            }

        # search leftside and collect internal index, left code is 0
        self.build_huffman_path(
            root.left,
            current_codes + [0],
            current_internal_indices + [root.internal_index],
            huffman_codes,
        )
        # search rightside and collect internal index, right code is 1
        self.build_huffman_path(
            root.right,
            current_codes + [1],
            current_internal_indices + [root.internal_index],
            huffman_codes,
        )

        return huffman_codes

    def forward(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            embedding (torch.Tensor): embedding vectors, shape(batch_size, 1 or neighbor_size, #subsequence, embed_dim)
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence)

        Returns:
            torch.Tensor: loss
        """
        # (N = B * c * #subseq), h
        embedding = embedding.view(-1, embedding.size(-1))

        # target mapping and concat
        # N
        paths = itemgetter(*target.flatten().tolist())(self.paths)

        # indices, shape is ?
        internal_indices = torch.cat([path["indices"] for path in paths])

        # index of sample
        # for example return 1, 1, 2, 3
        # if sample 1 with tree depth 2, sample 2 and 3 with tree depth 1
        sample_indices = []
        for i, path in enumerate(paths):
            sample_indices.extend(path["indices"].size(0) * [i])
        sample_indices = torch.LongTensor(sample_indices).to(self.context.device)

        # ?
        pred = torch.einsum(
            "xh, xh -> x",
            embedding[sample_indices],
            self.fc[internal_indices],
        )

        # ?
        target = torch.cat([path["code"] for path in paths])

        return binary_cross_entropy_with_logits(pred, target)
