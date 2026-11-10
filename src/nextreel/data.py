"""Load ratings, sort by time, remap ids and make the leave-one-out split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]
YEAR_PATTERN = r"\s*\((\d{4})\)\s*$"


def read_dat(path, names):
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


def load_ratings(data_dir):
    return read_dat(Path(data_dir) / "ratings.dat", RATING_COLUMNS)


def load_movies(data_dir):
    movies = read_dat(Path(data_dir) / "movies.dat", MOVIE_COLUMNS)
    year = movies["title"].str.extract(YEAR_PATTERN, expand=False)
    return movies.assign(
        name=movies["title"].str.replace(YEAR_PATTERN, "", regex=True).str.strip(),
        year=pd.to_numeric(year, errors="coerce").astype("Int64"),
        genres=movies["genres"].str.split("|"),
    )


def filter_min_interactions(ratings, min_count=5):
    while True:
        user_counts = ratings["user_id"].map(ratings["user_id"].value_counts())
        movie_counts = ratings["movie_id"].map(ratings["movie_id"].value_counts())
        keep = (user_counts >= min_count) & (movie_counts >= min_count)
        if keep.all():
            return ratings.reset_index(drop=True)
        ratings = ratings[keep]


def sort_by_time(ratings):
    return ratings.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def remap_ids(ratings):
    user_ids = np.sort(ratings["user_id"].unique())
    movie_ids = np.sort(ratings["movie_id"].unique())
    user_map = {int(u): i for i, u in enumerate(user_ids)}
    item_map = {int(m): i + 1 for i, m in enumerate(movie_ids)}
    ratings = ratings.assign(
        user=ratings["user_id"].map(user_map),
        item=ratings["movie_id"].map(item_map),
    )
    return ratings, user_map, item_map


def build_sequences(ratings):
    return ratings.groupby("user", sort=True)["item"].agg(list).tolist()


@dataclass
class Split:
    train: list
    valid: list
    test: list

    def histories(self, stage):
        if stage == "valid":
            return self.train, self.valid
        if stage == "test":
            return [seq + [v] for seq, v in zip(self.train, self.valid)], self.test
        raise ValueError(f"unknown stage {stage!r}")


def leave_one_out(sequences):
    train, valid, test = [], [], []
    for seq in sequences:
        if len(seq) < 3:
            raise ValueError("every user needs at least 3 interactions")
        train.append(list(seq[:-2]))
        valid.append(seq[-2])
        test.append(seq[-1])
    return Split(train, valid, test)
