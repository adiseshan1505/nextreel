"""SASRec: causal self-attention over movie ids."""

import torch
import torch.nn.functional as F
from torch import nn

from .attention import TransformerBlock


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=200, d=64, n_layers=2, n_heads=2, dropout=0.2,
                 use_position=True):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.use_position = use_position
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(d, n_heads, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def embed(self, seq):
        x = self.item_emb(seq)
        if self.use_position:
            positions = torch.arange(seq.size(1), device=seq.device)
            x = x + self.pos_emb(positions)
        return self.emb_dropout(x) * (seq > 0).unsqueeze(-1)

    def encode(self, seq, return_attention=False):
        pad_mask = seq > 0
        x = self.embed(seq)
        attentions = []
        for block in self.blocks:
            x, weights = block(x, pad_mask)
            attentions.append(weights)
        x = self.norm(x)
        return (x, attentions) if return_attention else x

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
