import torch
from ..models.relation.model import RelationNet


QUERY_SIZE = 2
INPUT_CHANNEL = 1
IMAGE_SIZE = 28
HIDDEN_DIM = 64
N_WAY = 5
K_SHOT = 5
NUM_CLASS = 20


class TestModel:
    def test_relation_net_feature_extract_shape(self):
        q = torch.rand(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = RelationNet(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        )

        y = m.feature_extract(q.view(-1, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        assert y.shape == torch.Size(
            (
                N_WAY * QUERY_SIZE,
                HIDDEN_DIM,
                IMAGE_SIZE // 4,
                IMAGE_SIZE // 4,
            )
        )

    def test_relation_net_get_logit_shape(self):
        q = torch.rand(N_WAY * QUERY_SIZE, HIDDEN_DIM, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        support = torch.rand(N_WAY, HIDDEN_DIM, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        m = RelationNet(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        )

        y = m.get_logit(q, support)

        assert y.shape == torch.Size((N_WAY * QUERY_SIZE, N_WAY))

    def test_relation_net_forward_shape(self):
        q = torch.rand(N_WAY, QUERY_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        support = torch.rand(N_WAY, K_SHOT, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        m = RelationNet(
            INPUT_CHANNEL,
            HIDDEN_DIM,
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        )

        y = m(q, support)

        assert y.shape == torch.Size((QUERY_SIZE, N_WAY, N_WAY))
