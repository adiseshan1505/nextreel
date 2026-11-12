"""Streamlit demo: pick 5 movies, get a top 10."""

import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nextreel.data import load_movielens
from nextreel.dataset import sequence_batch
from nextreel.explain import attention_rollout, because_you_watched, explanation_text
from nextreel.models import Popularity
from nextreel.train import load_model

DATA_DIR = Path(os.environ.get("NEXTREEL_DATA_DIR", ROOT / "data" / "ml-1m"))
RUNS_DIR = Path(os.environ.get("NEXTREEL_RUNS_DIR", ROOT / "runs"))
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
N_PICKS = 5
TOP_K = 10


@st.cache_resource(show_spinner="Downloading MovieLens 1M from GroupLens...")
def ensure_data():
    if DATA_DIR.exists():
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR.parent / "ml-1m.zip"
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA_DIR.parent)
    archive.unlink()


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


def sequence_scores(model, picks):
    seq = sequence_batch([picks], [0], model.max_len)
    return model.predict(seq)[0].numpy()


def mf_scores(model, picks):
    with torch.no_grad():
        user = model.item_emb.weight[list(picks)].mean(0)
        return (model.item_emb.weight @ user).numpy()


def shuffle_picks():
    picks = list(st.session_state.picks)
    shuffled = picks[:]
    while len(picks) > 1 and shuffled == picks:
        random.shuffle(shuffled)
    st.session_state.picks = shuffled
    st.session_state.previous = st.session_state.get("current")


def movement(item, rank, previous):
    if previous is None:
        return ""
    if item not in previous:
        return " *(new)*"
    delta = previous.index(item) + 1 - rank
    if delta > 0:
        return f" ↑{delta}"
    if delta < 0:
        return f" ↓{-delta}"
    return ""


def show_list(column, heading, items, previous=None):
    column.subheader(heading)
    column.markdown("\n".join(
        f"{rank}. {ml.title(item)}{movement(item, rank, previous)}"
        for rank, item in enumerate(items, 1)
    ))


st.set_page_config(page_title="NextReel", layout="wide")
st.title("NextReel")
st.caption("Pick the movies you watched, in order, and get the next 10.")

try:
    ensure_data()
except OSError as error:
    st.error(f"Could not download MovieLens 1M into {DATA_DIR}: {error}")
    st.stop()
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
st.button("Shuffle the order", on_click=shuffle_picks)

sasrec = load_trained("sasrec")
mf = load_trained("mf_bpr")
popular_col, mf_col, sasrec_col = st.columns(3)

show_list(popular_col, "Most popular", top_k(load_popularity().scores, picks))

if mf is None:
    mf_col.subheader("Matrix factorisation")
    mf_col.info("Train it with `python -m nextreel.train --config configs/mf_bpr.yaml`.")
else:
    show_list(mf_col, "Matrix factorisation", top_k(mf_scores(mf, picks), picks))
    mf_col.caption("Your vector is the average of your picks' vectors, so order is ignored.")

if sasrec is None:
    sasrec_col.subheader("NextReel")
    sasrec_col.info("Train it with `python -m nextreel.train --config configs/sasrec.yaml`.")
    st.stop()

current = top_k(sequence_scores(sasrec, picks), picks)
previous = st.session_state.pop("previous", None)
show_list(sasrec_col, "NextReel", current, previous)
st.session_state.current = current
if previous is not None:
    st.caption("Same movies, new order: arrows show how NextReel's top 10 moved.")
sasrec_col.caption(explanation_text(because_you_watched(sasrec, picks, top_n=1), ml.title))

st.subheader("Where NextReel looked")
weights = attention_rollout(sasrec, picks)
st.dataframe(
    pd.DataFrame({"Movie": [ml.title(i) for i in picks], "Attention": weights}),
    hide_index=True,
    column_config={
        "Attention": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
    },
)
st.caption("Attention shows where the model looked, not exactly why it chose a movie.")
