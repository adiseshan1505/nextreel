"""Load ratings, sort by time, remap ids and make the leave-one-out split."""

from pathlib import Path

import pandas as pd

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]


def read_dat(path, names):
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


def load_ratings(data_dir):
    return read_dat(Path(data_dir) / "ratings.dat", RATING_COLUMNS)


def load_movies(data_dir):
    return read_dat(Path(data_dir) / "movies.dat", MOVIE_COLUMNS)


def filter_min_interactions(ratings, min_count=5):
    while True:
        user_counts = ratings["user_id"].map(ratings["user_id"].value_counts())
        movie_counts = ratings["movie_id"].map(ratings["movie_id"].value_counts())
        keep = (user_counts >= min_count) & (movie_counts >= min_count)
        if keep.all():
            return ratings.reset_index(drop=True)
        ratings = ratings[keep]
