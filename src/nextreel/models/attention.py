"""Hand-written attention block that returns attention weights."""

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        if d % n_heads:
            raise ValueError("d must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        b, length, d = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        eye = torch.eye(length, dtype=torch.bool, device=x.device)
        allowed = allowed & (pad_mask[:, None, None, :] | eye)
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        h = self.dropout(weights) @ v
        h = h.transpose(1, 2).reshape(b, length, d)
        return self.out(h), weights
