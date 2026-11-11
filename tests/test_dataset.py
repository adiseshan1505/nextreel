import torch

from nextreel.dataset import TrainSequences, pad_left, sequence_batch, train_loader


def test_pad_left_keeps_the_most_recent_items():
    assert pad_left([1, 2, 3], 5).tolist() == [0, 0, 1, 2, 3]
    assert pad_left([1, 2, 3, 4, 5, 6], 4).tolist() == [3, 4, 5, 6]
    assert pad_left([], 3).tolist() == [0, 0, 0]


def test_targets_are_inputs_shifted_by_one():
    inputs, targets = TrainSequences([[5, 6, 7, 8]], 6)[0]
    assert inputs.tolist() == [0, 0, 0, 5, 6, 7]
    assert targets.tolist() == [0, 0, 0, 6, 7, 8]


def test_single_item_sequences_are_skipped():
    assert len(TrainSequences([[1], [1, 2]], 4)) == 1


def test_sequence_batch_picks_users_in_order():
    batch = sequence_batch([[1, 2], [3], [4, 5, 6]], [2, 0], 3)
    assert batch.tolist() == [[4, 5, 6], [0, 1, 2]]
    assert batch.dtype == torch.int64


def test_loader_is_reproducible():
    sequences = [[i, i + 1, i + 2] for i in range(1, 30)]
    first = [x.tolist() for x, _ in train_loader(sequences, 4, 8, seed=3)]
    second = [x.tolist() for x, _ in train_loader(sequences, 4, 8, seed=3)]
    assert first == second
