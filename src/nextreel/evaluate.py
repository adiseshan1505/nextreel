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
