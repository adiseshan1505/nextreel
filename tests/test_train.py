import json

import numpy as np
import pytest
import torch

from nextreel.data import load_movielens
from nextreel.train import DEFAULTS, build_model, load_model, main, merge, run


@pytest.fixture
def toy_data(tmp_path):
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "ml"
    data_dir.mkdir()
    ratings = []
    for user in range(1, 31):
        start = rng.integers(1, 15)
        for step in range(8):
            movie = (start + step) % 20 + 1
            ratings.append(f"{user}::{movie}::4::{1000 + step}")
    (data_dir / "ratings.dat").write_text("\n".join(ratings) + "\n")
    movies = [f"{m}::Movie {m} ({1990 + m})::Drama|Comedy" for m in range(1, 21)]
    (data_dir / "movies.dat").write_text("\n".join(movies) + "\n", encoding="latin-1")
    return data_dir


def toy_config(data_dir, out_dir, name):
    return merge(DEFAULTS, {
        "name": name,
        "data": {"dir": str(data_dir), "min_interactions": 2},
        "model": {"name": name, "max_len": 10, "d": 16, "dropout": 0.0},
        "training": {"batch_size": 8, "max_epochs": 2, "lr": 0.01},
        "evaluation": {"n_negatives": 5},
        "output": {"dir": str(out_dir)},
    })


def test_merge_keeps_unset_defaults():
    cfg = merge(DEFAULTS, {"model": {"name": "gru"}})
    assert cfg["model"]["name"] == "gru"
    assert cfg["model"]["d"] == DEFAULTS["model"]["d"]
    assert DEFAULTS["model"]["name"] == "sasrec"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model(merge(DEFAULTS, {"model": {"name": "nope"}}), 1, 1)


@pytest.mark.parametrize("name", ["sasrec", "gru", "mf_bpr", "popularity", "item2item"])
def test_every_model_runs_end_to_end(toy_data, tmp_path, name):
    cfg = toy_config(toy_data, tmp_path / "runs", name)
    ml = load_movielens(toy_data, 2)
    result = run(cfg, ml, seed=0, log=lambda _: None)
    for metric in ("hr@10", "ndcg@10", "sampled_hr@10", "coverage@10", "popularity@10"):
        assert 0.0 <= result["test"][metric] <= 1.0
    out = tmp_path / "runs" / name / "seed0"
    assert json.loads((out / "metrics.json").read_text())["model"] == name
    if name in ("sasrec", "gru", "mf_bpr"):
        model, _ = load_model(out / "model.pt")
        assert isinstance(model, torch.nn.Module)
        assert len(result["history"]) >= 1


def test_cli_writes_one_run_per_seed(toy_data, tmp_path):
    config = tmp_path / "pop.yaml"
    config.write_text(
        "model:\n  name: popularity\n"
        "data:\n  min_interactions: 2\n"
        "evaluation:\n  n_negatives: 5\n"
    )
    main(["--config", str(config), "--seeds", "0", "1",
          "--data-dir", str(toy_data), "--out-dir", str(tmp_path / "runs")])
    assert sorted(p.name for p in (tmp_path / "runs" / "pop").iterdir()) == ["seed0", "seed1"]
