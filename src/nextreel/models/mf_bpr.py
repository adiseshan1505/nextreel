"""BPR matrix factorisation baseline."""

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
