import torch

# import torch.nn.functional as F
from torch import nn

# from torch_scatter import scatter_logsumexp

from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import HuffmanNode


class HierarchicalSoftmax(nn.Module):
    def __init__(
        self,
        root: HuffmanNode,
        vocab_size: int,
        embed_dim: int,
        context: NLPContext | None = None,
    ):
        """hierarchical softmax loss

        Args:
            root (HuffmanNode): the root of a Huffman tree based on word counts
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context = context

        left_child, right_child, is_leaf, values = self.flatten_tree(root)
        self.register_buffer("left_child", torch.tensor(left_child))
        self.register_buffer("right_child", torch.tensor(right_child))
        self.register_buffer("is_leaf", torch.tensor(is_leaf, dtype=torch.bool))
        self.register_buffer("values", torch.tensor(values, dtype=torch.long))

        # self.path_indices, self.path_codes = self.get_paths(root, vocab_size)

        # vectors of path nodes
        self.node_embedding = nn.Embedding(vocab_size - 1, embed_dim)

    @staticmethod
    def flatten_tree(
        root: HuffmanNode,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        left_child = [-1]
        right_child = [-1]
        is_leaf = [root.is_leaf]
        values = [root.value]

        # bfs
        nodes = [root]
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if not n.is_leaf:
                left_idx = len(nodes)
                right_idx = len(nodes) + 1
                nodes.append(n.left)
                nodes.append(n.right)

                left_child[i] = left_idx
                right_child[i] = right_idx
                left_child.append(-1)
                right_child.append(-1)
                left_child.append(-1)
                right_child.append(-1)

                is_leaf.append(n.left.is_leaf)
                is_leaf.append(n.right.is_leaf)
                values.append(n.left.value)
                values.append(n.right.value)
            i += 1

        return left_child, right_child, is_leaf, values

    # @staticmethod
    # def get_paths(
    #     root: HuffmanNode,
    #     vocab_size: int,
    # ) -> tuple[dict[int, torch.LongTensor], dict[int, torch.Tensor]]:
    #     """get paths to leaves of huffman tree

    #     Args:
    #         root (HuffmanNode): root of huffman tree
    #         vocab_size (int): vocabulary size

    #     Returns:
    #         tuple[dict[int, torch.LongTensor], dict[int, torch.Tensor]]: intermediate indices to leaves and codes
    #     """
    #     assert root is not None, "root should be a HuffmanNode"
    #     N = vocab_size

    #     # shape is depth
    #     path_indices = {}
    #     # shape is depth; value is 0 or 1
    #     path_codes = {}

    #     def traverse(node: HuffmanNode, indices: list[int], codes: list[int]):
    #         if node.is_leaf:
    #             leaf_idx = node.value
    #             path_indices[leaf_idx] = torch.LongTensor(indices)
    #             path_codes[leaf_idx] = torch.FloatTensor(codes)
    #             return

    #         node_idx = node.value - N
    #         if node.left:
    #             traverse(node.left, indices + [node_idx], codes + [0])
    #         if node.right:
    #             traverse(node.right, indices + [node_idx], codes + [1])

    #     traverse(root, [], [])

    #     return path_indices, path_codes

    # def get_logit(self, x: torch.Tensor) -> torch.Tensor:
    #     """forward

    #     Args:
    #         x (torch.Tensor): query vectors, shape(..., embed_dim)

    #     Returns:
    #         torch.Tensor: logits over vocab
    #     """
    #     other_shape = x.shape
    #     # n, h
    #     x = x.view(-1, self.embed_dim)

    #     # cache intermediate logit
    #     # n, v-1
    #     y = F.linear(x, self.node_embedding.weight)

    #     # TODO: sort indices
    #     # TODO: too large
    #     # get path indices
    #     # n, v^2
    #     y = y[:, torch.cat(tuple(self.path_indices.values()))]
    #     y = torch.where(torch.cat(tuple(self.path_codes.values())), y, -y)

    #     # scatter log sum exp
    #     y = scatter_logsumexp(
    #         y,
    #         torch.range(
    #             self.vocab_size,
    #             device=self.context.device,
    #         ).repeat_interleave(tuple(v.size(0) for v in self.path_indices.values())),
    #         dim=1,
    #     )

    #     return y.view(*other_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): query vectors, shape(..., embed_dim)

        Returns:
            torch.Tensor: prediction
        """
        other_shape = x.shape[:-1]
        # n, h
        x = x.view(-1, self.embed_dim)
        N = x.shape[0]

        node = torch.zeros(N, dtype=torch.long, device=self.context.device)
        done = self.is_leaf[node].clone()
        # in case root itself is a leaf (edge case)
        result = torch.where(done, self.values[node], torch.full_like(node, -1))

        while not done.all().item():
            not_done = ~done
            remained_indices = node[not_done]
            # n, h
            z = self.node_embedding(self.values[remained_indices] - self.vocab_size)
            # n
            logit = torch.einsum("xh,xh->x", z, x[not_done])
            node[not_done] = torch.where(
                logit > 0,
                self.right_child[remained_indices],
                self.left_child[remained_indices],
            )

            reach_leaf = self.is_leaf[node[not_done]]
            result[not_done] = torch.where(
                reach_leaf,
                self.values[node[not_done]],
                result[not_done],
            )
            done[not_done] = reach_leaf

        # (...,)
        return result.view(*other_shape)
