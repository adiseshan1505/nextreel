"""Metrics checked on hand-made toy rankings."""

import math

import numpy as np
import pytest
import torch

from nextreel.evaluate import (
    Evaluator,
    coverage,
    hit_rate,
    mask_history,
    ndcg,
    popularity_bias,
    popularity_percentile,
    sample_negatives,
    sampled_ranks,
    target_ranks,
)


def test_hit_rate_counts_ranks_inside_k():
    assert hit_rate([1, 5, 10, 11], k=10) == 0.75
    assert hit_rate([11, 50], k=10) == 0.0


def test_ndcg_by_hand():
    assert ndcg([1]) == 1.0
    assert ndcg([3]) == pytest.approx(0.5)
    assert ndcg([11]) == 0.0
    assert ndcg([1, 3, 11, 2]) == pytest.approx((1 + 0.5 + 0 + 1 / math.log2(3)) / 4)


def test_rank_of_target_in_a_known_order():
    scores = torch.tensor([[0.0, 0.9, 0.1, 0.5, 0.7], [0.0, 0.1, 0.2, 0.3, 0.4]])
    targets = torch.tensor([3, 4])
    assert target_ranks(scores, targets).tolist() == [3, 1]


def test_watched_movies_and_padding_are_removed():
    scores = torch.tensor([[9.0, 0.9, 0.1, 0.5, 0.7]])
    masked = mask_history(scores, [[1, 4]], torch.tensor([3]))
    assert target_ranks(masked, torch.tensor([3])).tolist() == [1]
    assert torch.isinf(masked[0, [0, 1, 4]]).all()


def test_target_is_kept_even_if_it_was_watched_before():
    scores = torch.tensor([[0.0, 0.9, 0.1, 0.5]])
    masked = mask_history(scores, [[1, 3]], torch.tensor([3]))
    assert masked[0, 3] == 0.5


def test_ties_are_broken_at_random_not_in_the_targets_favour():
    scores = torch.zeros(2000, 11)
    targets = torch.full((2000,), 5)
    ranks = target_ranks(scores, targets, torch.Generator().manual_seed(0))
    assert ranks.min() == 1 and ranks.max() == 11
    assert ranks.mean() == pytest.approx(6, abs=0.3)


def test_sampled_rank_only_counts_the_negatives():
    scores = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    ranks = sampled_ranks(scores, torch.tensor([3]), torch.tensor([[1, 2, 6]]))
    assert ranks.tolist() == [2]


def test_negatives_skip_history_and_target():
    rng = np.random.default_rng(0)
    negatives = sample_negatives([[1, 2, 3]], [4], 10, 6, rng)
    assert sorted(negatives[0]) == [5, 6, 7, 8, 9, 10]


def test_coverage_and_popularity():
    recommended = np.array([[1, 2], [1, 3]])
    assert coverage(recommended, 6) == 0.5
    percentile = popularity_percentile(np.array([0, 10, 5, 1, 1]))
    assert percentile.tolist() == [0.0, 0.75, 0.5, 0.0, 0.0]
    assert popularity_bias(np.array([[1, 2]]), percentile) == pytest.approx(0.625)


def test_evaluator_on_a_perfect_scorer():
    histories = [[1, 2], [3], [2, 4]]
    targets = [5, 1, 3]
    counts = np.array([0, 3, 2, 2, 1, 1, 1, 1])

    def oracle(users):
        scores = torch.zeros(len(users), 8)
        scores[torch.arange(len(users)), torch.as_tensor([targets[u] for u in users])] = 1.0
        return scores

    metrics = Evaluator(histories, targets, counts, k=2, n_negatives=3).evaluate(oracle, batch_size=2)
    assert metrics["hr@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0
    assert metrics["sampled_hr@2"] == 1.0
