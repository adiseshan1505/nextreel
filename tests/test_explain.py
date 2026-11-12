import numpy as np
import pytest
import torch

from nextreel.explain import (
    attention_rollout,
    because_you_watched,
    explanation_text,
    last_step_attention,
)
from nextreel.models import SASRec


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SASRec(n_items=20, max_len=6, d=16, n_layers=2, n_heads=2, dropout=0.0).eval()


@pytest.mark.parametrize("weigh", [last_step_attention, attention_rollout])
def test_weights_cover_the_history_and_sum_to_one(model, weigh):
    weights = weigh(model, [3, 7, 9])
    assert weights.shape == (3,)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)


def test_long_histories_are_cut_to_max_len(model):
    weights = attention_rollout(model, list(range(1, 11)))
    assert weights.shape == (6,)


def test_reasons_come_from_the_history(model):
    history = [3, 7, 9, 12]
    reasons = because_you_watched(model, history, top_n=2)
    assert len(reasons) == 2
    assert all(item in history for item, _ in reasons)
    assert reasons[0][1] >= reasons[1][1]


def test_unknown_method_is_rejected(model):
    with pytest.raises(ValueError):
        because_you_watched(model, [1, 2], method="magic")


def test_explanation_text():
    titles = {1: "Toy Story", 2: "Heat", 3: "Fargo"}.get
    assert explanation_text([], titles) == ""
    assert explanation_text([(1, 0.9)], titles) == "Because you watched Toy Story"
    assert explanation_text([(1, 0.5), (2, 0.3), (3, 0.2)], titles) == (
        "Because you watched Toy Story, Heat and Fargo"
    )
