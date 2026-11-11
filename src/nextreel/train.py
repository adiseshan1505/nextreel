"""Training loop with early stopping on validation NDCG@10."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import load_movielens
from .dataset import sequence_batch, train_loader
from .evaluate import Evaluator
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


def fit(model, ml, cfg, evaluator, seed=0, device="cpu", log=print):
    t, k = cfg["training"], cfg["evaluation"]["k"]
    if isinstance(model, (Popularity, ItemToItem)):
        model.fit(ml.split.train)
        return {"best_epoch": 0, "best_valid": None, "history": []}
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    max_len = cfg["model"]["max_len"]
    if isinstance(model, BPRMF):
        sampler = TripletSampler(ml.split.train, ml.n_items, seed)

        def step():
            return train_mf_epoch(model, sampler, t["batch_size"], optimiser, device)
    else:
        loader = train_loader(ml.split.train, max_len, t["batch_size"], seed)

        def step():
            return train_sequence_epoch(model, loader, optimiser, device)

    histories, _ = ml.split.histories("valid")
    key = f"ndcg@{k}"
    best_score, best_epoch, best_state, waited, history = -1.0, 0, None, 0, []
    for epoch in range(1, t["max_epochs"] + 1):
        start = time.time()
        loss = step()
        model.eval()
        scorer = make_scorer(model, histories, max_len, device)
        metrics = evaluator.evaluate(scorer, cfg["evaluation"]["batch_size"], sampled=False)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        log(f"epoch {epoch:3d}  loss {loss:.4f}  valid {key} {metrics[key]:.4f}  {time.time() - start:.0f}s")
        if metrics[key] > best_score:
            best_score, best_epoch, waited = metrics[key], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= t["early_stopping_patience"]:
                log(f"early stop, best epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_valid": best_score, "history": history}


def build_evaluators(ml, cfg):
    e = cfg["evaluation"]
    counts = ml.item_counts()
    return {
        stage: Evaluator(*ml.split.histories(stage), counts, e["k"], e["n_negatives"], e["seed"])
        for stage in ("valid", "test")
    }


def save_run(out_dir, model, cfg, ml, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if isinstance(model, torch.nn.Module):
        torch.save(
            {
                "config": cfg,
                "n_users": ml.n_users,
                "n_items": ml.n_items,
                "state_dict": model.cpu().state_dict(),
            },
            out_dir / "model.pt",
        )


def run(cfg, ml, seed, device="cpu", log=print):
    set_seed(seed)
    evaluators = build_evaluators(ml, cfg)
    model = build_model(cfg, ml.n_users, ml.n_items)
    summary = fit(model, ml, cfg, evaluators["valid"], seed, device, log)
    max_len, batch_size = cfg["model"]["max_len"], cfg["evaluation"]["batch_size"]
    scores = {}
    for stage, evaluator in evaluators.items():
        histories, _ = ml.split.histories(stage)
        scores[stage] = evaluator.evaluate(make_scorer(model, histories, max_len, device), batch_size)
    result = {
        "name": cfg["name"],
        "model": cfg["model"]["name"],
        "seed": seed,
        **summary,
        "valid": scores["valid"],
        "test": scores["test"],
    }
    save_run(Path(cfg["output"]["dir"]) / cfg["name"] / f"seed{seed}", model, cfg, ml, result)
    return result
