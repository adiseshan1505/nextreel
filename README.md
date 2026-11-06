## NextReel- pays attention to what you watch

So NextReel is a SASRec movie recommender that predicts the next movie a user will watch from the order of their watch history. It uses a transformer built from scratch in PyTorch and trains on a laptop CPU.
<hr>
Given all the movies that user has watched, in order, rank every movie in the catalog by how likely it is to be the next one they watch. 

$$\hat{s}_{n+1} = \arg\max_{i \in \mathrm{I} \setminus S_u} P\left(i \mid s_1, s_2, \ldots, s_n\right)$$
In the above equation S= s1,s2,...,sn are like the user u's watch history in time order, and the catalog minus already-watched movies in the candidate set.

### Project structure

These files are only skeletons for now.

```
configs/        model and training settings
src/nextreel/   data pipeline, models, training, evaluation and explanations
app/            Streamlit demo
tests/          split and metric tests
results/        results table
```
