## NextReel- pays attention to what you watch

So NextReel is a SASRec movie recommender that predicts the next movie a user will watch from the order of their watch history. It uses a transformer built from scratch in PyTorch and trains on a laptop CPU.
<hr>
Given all the movies that user has watched, in order, rank every movie in the catalog by how likely it is to be the next one they watch. 

```math
\hat{s}_{n+1} = \arg\max_{i \in \mathrm{I} \setminus S_u} P\left(i \mid s_1, s_2, \ldots, s_n\right)
```

In the above equation S= s1,s2,...,sn are like the user u's watch history in time order, and the catalog minus already-watched movies in the candidate set.

### Dataset

NextReel uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): 1,000,209 ratings of about 3,900 movies by 6,040 users.

| File | Format | Used for |
| --- | --- | --- |
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` | The watch sequences |
| `movies.dat` | `MovieID::Title::Genres` | Movie titles, genres and release years |

Download the zip from GroupLens and unzip it into `data/`. The dataset may not be redistributed, so `data/` is never committed.

**Preprocessing.** Every rating counts as a "watch", whatever the star value. Each user's ratings are sorted by timestamp, and users and movies with fewer than 5 interactions are dropped.

**Split.** Leave-one-out by time: each user's last movie is the test target, the second-to-last is the validation target, and everything before that is used for training.

**Limitation.** MovieLens timestamps record when a user *rated* a movie, not when they watched it, and many users rated dozens in one sitting. So "order" here is partly rating-session order.

The dataset is for research use only. If you use it, cite F. Maxwell Harper and Joseph A. Konstan, "The MovieLens Datasets: History and Context", *ACM TiiS* 5(4), 2015.

### Project structure

These files are only skeletons for now.

```
configs/        model and training settings
src/nextreel/   data pipeline, models, training, evaluation and explanations
app/            Streamlit demo
tests/          split and metric tests
results/        results table
```
