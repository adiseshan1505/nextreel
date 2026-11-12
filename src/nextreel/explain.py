"""Attention-based "because you watched" explanations."""

import numpy as np
import torch

from .dataset import pad_left


def encode_history(model, history):
    seq = torch.from_numpy(pad_left(history, model.max_len))[None]
    with torch.no_grad():
        _, attentions = model.encode(seq, return_attention=True)
    return attentions


def last_step_attention(model, history, layer=-1):
    attentions = encode_history(model, history)
    weights = attentions[layer][0, :, -1].mean(0)
    n = min(len(history), model.max_len)
    return weights[-n:].numpy()
