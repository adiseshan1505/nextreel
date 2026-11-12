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


def attention_rollout(model, history):
    attentions = encode_history(model, history)
    length = attentions[0].size(-1)
    eye = torch.eye(length)
    rollout = eye
    for weights in attentions:
        mixed = 0.5 * weights[0].mean(0) + 0.5 * eye
        rollout = (mixed / mixed.sum(-1, keepdim=True)) @ rollout
    n = min(len(history), model.max_len)
    return rollout[-1, -n:].numpy()
