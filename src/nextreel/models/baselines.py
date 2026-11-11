"""Popularity and item-to-item transition baselines."""

import numpy as np


def flatten(sequences):
    return np.concatenate([np.asarray(seq, dtype=np.int64) for seq in sequences])


class Popularity:
    def __init__(self, n_items):
        self.n_items = n_items
        self.scores = np.zeros(n_items + 1, dtype=np.float32)

    def fit(self, sequences):
        counts = np.bincount(flatten(sequences), minlength=self.n_items + 1)
        self.scores = counts.astype(np.float32)
        self.scores[0] = -np.inf
        return self

    def predict(self, histories):
        return np.tile(self.scores, (len(histories), 1))


class ItemToItem:
    def __init__(self, n_items, popularity_weight=1e-6):
        self.n_items = n_items
        self.popularity_weight = popularity_weight
        self.transitions = np.zeros((n_items + 1, n_items + 1), dtype=np.float32)
        self.popularity = np.zeros(n_items + 1, dtype=np.float32)

    def fit(self, sequences):
        prev = flatten(seq[:-1] for seq in sequences)
        nxt = flatten(seq[1:] for seq in sequences)
        self.transitions[:] = 0
        np.add.at(self.transitions, (prev, nxt), 1.0)
        counts = np.bincount(flatten(sequences), minlength=self.n_items + 1)
        self.popularity = (counts / max(counts.max(), 1)).astype(np.float32)
        return self

    def predict(self, histories):
        last = np.array([h[-1] if len(h) else 0 for h in histories])
        scores = self.transitions[last] + self.popularity_weight * self.popularity
        scores[:, 0] = -np.inf
        return scores
