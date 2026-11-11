import torch
import torch.nn.functional as F
from torch import nn


class GRURec(nn.Module):
    def __init__(self, n_items, max_len=200, d=64, n_layers=1, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d, d, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq):
        x = self.dropout(self.item_emb(seq))
        hidden, _ = self.gru(x)
        return self.norm(hidden)

    def loss(self, seq, targets):
        hidden = self.encode(seq)
        mask = targets > 0
        logits = hidden[mask] @ self.item_emb.weight[1:].T
        return F.cross_entropy(logits, targets[mask] - 1)

    @torch.no_grad()
    def predict(self, seq):
        hidden = self.encode(seq)[:, -1]
        scores = hidden @ self.item_emb.weight.T
        scores[:, 0] = float("-inf")
        return scores
