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


def because_you_watched(model, history, top_n=3, method="rollout"):
    if method == "rollout":
        weights = attention_rollout(model, history)
    elif method == "last":
        weights = last_step_attention(model, history)
    else:
        raise ValueError(f"unknown method {method!r}")
    items = list(history)[-len(weights):]
    order = np.argsort(-weights, kind="stable")[:top_n]
    return [(items[i], float(weights[i])) for i in order]


def explanation_text(reasons, title):
    if not reasons:
        return ""
    names = [title(item) for item, _ in reasons]
    if len(names) == 1:
        return f"Because you watched {names[0]}"
    return f"Because you watched {', '.join(names[:-1])} and {names[-1]}"
