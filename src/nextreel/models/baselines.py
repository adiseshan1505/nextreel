"""Popularity and item-to-item transition baselines."""

import numpy as np


class Popularity:
    def __init__(self, n_items):
        self.n_items = n_items
        self.scores = np.zeros(n_items + 1, dtype=np.float32)

    def fit(self, sequences):
        counts = np.bincount(np.concatenate(sequences), minlength=self.n_items + 1)
        self.scores = counts.astype(np.float32)
        self.scores[0] = -np.inf
        return self

    def predict(self, histories):
        return np.tile(self.scores, (len(histories), 1))
