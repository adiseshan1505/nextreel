"""SASRec: causal self-attention over movie ids."""

import torch
from torch import nn

from .attention import TransformerBlock


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=200, d=64, n_layers=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
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
        positions = torch.arange(seq.size(1), device=seq.device)
        x = self.item_emb(seq) + self.pos_emb(positions)
        return self.emb_dropout(x) * (seq > 0).unsqueeze(-1)

    def encode(self, seq):
        pad_mask = seq > 0
        x = self.embed(seq)
        for block in self.blocks:
            x, _ = block(x, pad_mask)
        return self.norm(x)
