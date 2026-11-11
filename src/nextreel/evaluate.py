"""Full-ranking HR@K, NDCG@K, coverage and popularity bias."""

import numpy as np
import torch


def hit_rate(ranks, k=10):
    return float(np.mean(np.asarray(ranks) <= k))


def ndcg(ranks, k=10):
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)))
