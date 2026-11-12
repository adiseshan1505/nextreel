"""Streamlit demo: pick 5 movies, get a top 10."""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nextreel.data import load_movielens
from nextreel.models import Popularity
from nextreel.train import load_model

DATA_DIR = Path(os.environ.get("NEXTREEL_DATA_DIR", ROOT / "data" / "ml-1m"))
RUNS_DIR = Path(os.environ.get("NEXTREEL_RUNS_DIR", ROOT / "runs"))
N_PICKS = 5
TOP_K = 10


@st.cache_resource
def load_data():
    return load_movielens(DATA_DIR)


@st.cache_resource
def load_trained(name):
    paths = sorted((RUNS_DIR / name).glob("seed*/model.pt"))
    if not paths:
        return None
    model, _ = load_model(paths[0])
    return model


@st.cache_resource
def load_popularity():
    ml = load_data()
    return Popularity(ml.n_items).fit(ml.split.train)


def top_k(scores, picks, k=TOP_K):
    scores = np.asarray(scores, dtype=np.float64).copy()
    scores[0] = -np.inf
    scores[list(picks)] = -np.inf
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


st.set_page_config(page_title="NextReel", layout="wide")
st.title("NextReel")
st.caption("Pick the movies you watched, in order, and get the next 10.")

if not DATA_DIR.exists():
    st.error(f"MovieLens 1M not found in {DATA_DIR}. See the README to download it.")
    st.stop()

ml = load_data()
counts = ml.item_counts()
options = [int(i) for i in np.argsort(-counts[1:], kind="stable") + 1]

picks = st.multiselect(
    f"Pick {N_PICKS} movies in the order you watched them",
    options,
    format_func=ml.title,
    max_selections=N_PICKS,
    key="picks",
)

if len(picks) < N_PICKS:
    st.info(f"Pick {N_PICKS - len(picks)} more.")
    st.stop()

st.markdown("**Watch order:** " + " → ".join(ml.title(i) for i in picks))
