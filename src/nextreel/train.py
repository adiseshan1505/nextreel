"""Training loop with early stopping on validation NDCG@10."""

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import sequence_batch, train_loader
from .models import BPRMF, GRURec, ItemToItem, Popularity, SASRec, TripletSampler

DEFAULTS = {
    "data": {"dir": "data/ml-1m", "min_interactions": 5},
    "model": {
        "name": "sasrec",
        "max_len": 200,
        "d": 64,
        "n_layers": 2,
        "n_heads": 2,
        "dropout": 0.2,
        "use_position": True,
    },
    "training": {
        "loss": "cross_entropy",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "max_epochs": 200,
        "early_stopping_patience": 5,
        "n_seeds": 3,
    },
    "evaluation": {"k": 10, "n_negatives": 100, "batch_size": 256, "seed": 0},
    "output": {"dir": "runs"},
}


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    with open(path) as f:
        cfg = merge(DEFAULTS, yaml.safe_load(f))
    cfg.setdefault("name", Path(path).stem)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, n_users, n_items):
    m = cfg["model"]
    name = m["name"]
    if name == "sasrec":
        return SASRec(
            n_items, m["max_len"], m["d"], m["n_layers"], m["n_heads"], m["dropout"],
            m.get("use_position", True),
        )
    if name == "gru":
        return GRURec(n_items, m["max_len"], m["d"], m["n_layers"], m["dropout"])
    if name == "mf_bpr":
        return BPRMF(n_users, n_items, m["d"])
    if name == "popularity":
        return Popularity(n_items)
    if name == "item2item":
        return ItemToItem(n_items)
    raise ValueError(f"unknown model {name!r}")


def make_scorer(model, histories, max_len, device="cpu"):
    if isinstance(model, (SASRec, GRURec)):
        def score(users):
            seq = sequence_batch(histories, users, max_len).to(device)
            return model.predict(seq).cpu()
    elif isinstance(model, BPRMF):
        def score(users):
            return model.predict(torch.as_tensor(users, device=device)).cpu()
    else:
        def score(users):
            return model.predict([histories[u] for u in users])
    return score


def train_sequence_epoch(model, loader, optimiser, device):
    model.train()
    total, batches = 0.0, 0
    for seq, targets in loader:
        optimiser.zero_grad()
        loss = model.loss(seq.to(device), targets.to(device))
        loss.backward()
        optimiser.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


def train_mf_epoch(model, sampler, batch_size, optimiser, device):
    model.train()
    users, positives, negatives = (torch.as_tensor(x) for x in sampler.sample())
    total, batches = 0.0, 0
    for start in range(0, len(users), batch_size):
        end = start + batch_size
        optimiser.zero_grad()
        loss = model.loss(
            users[start:end].to(device),
            positives[start:end].to(device),
            negatives[start:end].to(device),
        )
        loss.backward()
        optimiser.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)
