"""BPR matrix factorisation baseline."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, d=64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, users, items):
        return (self.user_emb(users) * self.item_emb(items)).sum(-1)

    def loss(self, users, positives, negatives):
        return -F.logsigmoid(self(users, positives) - self(users, negatives)).mean()

    @torch.no_grad()
    def predict(self, users):
        scores = self.user_emb(users) @ self.item_emb.weight.T
        scores[:, 0] = float("-inf")
        return scores


class TripletSampler:
    def __init__(self, sequences, n_items, seed=0):
        lengths = [len(seq) for seq in sequences]
        self.users = np.repeat(np.arange(len(sequences)), lengths)
        self.positives = np.concatenate([np.asarray(seq, dtype=np.int64) for seq in sequences])
        self.n_items = n_items
        self.seen = np.unique(self.users * (n_items + 1) + self.positives)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.positives)

    def sample(self):
        negatives = self.rng.integers(1, self.n_items + 1, size=len(self.positives))
        while True:
            clash = np.isin(self.users * (self.n_items + 1) + negatives, self.seen)
            if not clash.any():
                break
            negatives[clash] = self.rng.integers(1, self.n_items + 1, size=int(clash.sum()))
        order = self.rng.permutation(len(self.positives))
        return self.users[order], self.positives[order], negatives[order]
