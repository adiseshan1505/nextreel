"""No test or validation movie leaks into training sequences."""

from pathlib import Path

import pandas as pd
import pytest

from nextreel.data import (
    build_sequences,
    filter_min_interactions,
    leave_one_out,
    load_movielens,
    remap_ids,
    sort_by_time,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-1m"


def toy_ratings():
    rows = []
    for user in range(1, 5):
        for step, movie in enumerate(range(10, 16)):
            rows.append((user, movie + user, 4, 1000 * user - step))
    rows.append((9, 99, 5, 1))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def toy_split(min_count=2):
    ratings = filter_min_interactions(toy_ratings(), min_count)
    ratings, _, item_map = remap_ids(sort_by_time(ratings))
    return ratings, item_map, leave_one_out(build_sequences(ratings))


def test_filter_drops_rare_users_and_movies():
    ratings = filter_min_interactions(toy_ratings(), 2)
    assert 9 not in set(ratings["user_id"])
    assert 99 not in set(ratings["movie_id"])
    assert ratings.groupby("movie_id").size().min() >= 2


def test_sequences_follow_time_order():
    ratings, _, _ = toy_split()
    for _, group in ratings.groupby("user"):
        assert group["timestamp"].is_monotonic_increasing


def test_padding_id_is_never_used():
    _, item_map, split = toy_split()
    assert min(item_map.values()) == 1
    assert all(0 not in seq for seq in split.train)


def test_valid_and_test_are_the_last_two_items():
    ratings, _, split = toy_split()
    for user, group in ratings.groupby("user"):
        items = group["item"].tolist()
        assert split.train[user] == items[:-2]
        assert split.valid[user] == items[-2]
        assert split.test[user] == items[-1]


def test_stage_histories_hide_their_target():
    _, _, split = toy_split()
    for stage in ("valid", "test"):
        histories, targets = split.histories(stage)
        for history, target in zip(histories, targets):
            assert target not in history
    test_histories, _ = split.histories("test")
    assert all(h[-1] == v for h, v in zip(test_histories, split.valid))


def test_too_short_sequence_is_rejected():
    with pytest.raises(ValueError):
        leave_one_out([[1, 2]])


@pytest.mark.skipif(not DATA_DIR.exists(), reason="MovieLens 1M not downloaded")
def test_movielens_has_no_leakage():
    ml = load_movielens(DATA_DIR)
    assert ml.n_users == 6040
    for train, valid, test in zip(ml.split.train, ml.split.valid, ml.split.test):
        assert valid not in train
        assert test not in train
        assert test != valid
