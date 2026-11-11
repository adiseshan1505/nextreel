"""Full-ranking HR@K, NDCG@K, coverage and popularity bias."""

import numpy as np
import torch


def hit_rate(ranks, k=10):
    return float(np.mean(np.asarray(ranks) <= k))


def ndcg(ranks, k=10):
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)))


def mask_history(scores, histories, targets=None):
    scores = scores.clone()
    kept = None if targets is None else scores.gather(1, targets[:, None])
    rows = np.repeat(np.arange(len(histories)), [len(h) for h in histories])
    cols = np.concatenate([np.asarray(h, dtype=np.int64) for h in histories])
    scores[torch.as_tensor(rows), torch.as_tensor(cols)] = float("-inf")
    scores[:, 0] = float("-inf")
    if targets is not None:
        scores.scatter_(1, targets[:, None], kept)
    return scores


def target_ranks(scores, targets, generator=None):
    target_scores = scores.gather(1, targets[:, None])
    higher = (scores > target_scores).sum(1)
    ties = (scores == target_scores).sum(1) - 1
    offset = (torch.rand(len(targets), generator=generator) * (ties + 1)).floor().long()
    return (higher + 1 + offset).numpy()


def sample_negatives(histories, targets, n_items, n_negatives, rng):
    negatives = np.empty((len(targets), n_negatives), dtype=np.int64)
    for i, (history, target) in enumerate(zip(histories, targets)):
        seen = np.zeros(n_items + 1, dtype=bool)
        seen[0] = True
        seen[np.asarray(history, dtype=np.int64)] = True
        seen[target] = True
        negatives[i] = rng.choice(np.flatnonzero(~seen), n_negatives, replace=False)
    return negatives


def sampled_ranks(scores, targets, negatives, generator=None):
    candidates = torch.cat([targets[:, None], negatives], dim=1)
    first = torch.zeros(len(targets), dtype=torch.long)
    return target_ranks(scores.gather(1, candidates), first, generator)


def coverage(recommended, n_items):
    return len(np.unique(recommended)) / n_items


def popularity_percentile(item_counts):
    counts = np.asarray(item_counts[1:])
    percentile = np.searchsorted(np.sort(counts), counts, side="left") / len(counts)
    return np.concatenate([[0.0], percentile])


def popularity_bias(recommended, percentile):
    return float(np.mean(percentile[recommended]))
