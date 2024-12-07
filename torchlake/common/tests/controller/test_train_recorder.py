import pytest
import torch
from ...controller.recorder import TrainRecorder

BATCH_SIZE = 10
EPOCH = 10


def xy_data():
    for _ in range(BATCH_SIZE):
        yield [
            torch.rand(1, 5),
            torch.rand(1),
        ]


def x_data():
    for _ in range(BATCH_SIZE):
        yield torch.rand(1, 5)


class TestSuccess:

    def test_epoch(self):
        recorder = TrainRecorder(0, 10)

        for e in range(recorder.current_epoch, recorder.total_epoch):
            assert recorder.current_epoch == e
            recorder.increment_epoch()

        assert recorder.is_final_epoch()

    @pytest.mark.parametrize("data", [xy_data(), x_data()])
    def test_calc_dataset_size(self, data):
        recorder = TrainRecorder()

        count = recorder.calc_dataset_size(data)

        assert BATCH_SIZE == count

    @pytest.mark.parametrize("num_loss,losses", [(1, [3]), (3, [1, 2, 3])])
    def test_running_loss(self, num_loss, losses):
        recorder = TrainRecorder(num_loss=num_loss)

        recorder.increment_running_loss(*losses)

        assert losses == recorder.running_losses

    @pytest.mark.parametrize("num_loss,losses", [(1, [3.0]), (3, [1.0, 2.0, 3.0])])
    def test_training_loss(self, num_loss, losses):
        recorder = TrainRecorder(num_loss=num_loss)

        for _ in range(EPOCH):
            recorder.reset_running_loss()
            recorder.increment_running_loss(*losses)
            recorder.enqueue_training_loss()

        assert [[loss] * EPOCH for loss in losses] == recorder.training_losses

    @pytest.mark.parametrize(
        "num_loss,losses,expected",
        [
            (1, [[3.0], [2.0]], [-33]),
            (3, [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], [-50, -50, -50]),
        ],
    )
    def test_get_last_improvement(self, num_loss, losses, expected: list[int]):
        recorder = TrainRecorder(num_loss=num_loss)

        for loss in losses:
            recorder.reset_running_loss()
            recorder.increment_running_loss(*loss)
            recorder.enqueue_training_loss()

        output = recorder.get_last_improvement()

        assert expected == output
