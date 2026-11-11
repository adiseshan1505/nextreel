import numpy as np
import torch

from nextreel.models import BPRMF, GRURec, ItemToItem, Popularity, TripletSampler

SEQUENCES = [[1, 2, 3], [1, 2, 4], [2, 3, 1], [5, 1, 2]]


def test_popularity_ranks_by_count():
    scores = Popularity(5).fit(SEQUENCES).predict([[1], [3]])
    assert scores.shape == (2, 6)
    assert np.isneginf(scores[:, 0]).all()
    assert list(np.argsort(-scores[0])[:2]) == [1, 2]


def test_item_to_item_follows_the_last_movie():
    model = ItemToItem(5).fit(SEQUENCES)
    assert model.transitions[1, 2] == 3
    assert model.transitions[2, 3] == 2
    scores = model.predict([[4, 1], [5, 2]])
    assert scores[0].argmax() == 2
    assert scores[1].argmax() == 3


def test_item_to_item_falls_back_to_popularity():
    scores = ItemToItem(5).fit(SEQUENCES).predict([[4]])
    assert scores[0].argmax() in (1, 2)


def test_triplet_negatives_are_never_watched():
    sampler = TripletSampler(SEQUENCES, 6, seed=0)
    for _ in range(5):
        users, positives, negatives = sampler.sample()
        assert len(users) == len(sampler) == 12
        for user, negative in zip(users, negatives):
            assert negative not in SEQUENCES[user]
        assert sorted(zip(users, positives)) == sorted(
            (u, i) for u, seq in enumerate(SEQUENCES) for i in seq
        )


def test_bpr_learns_to_prefer_watched_movies():
    torch.manual_seed(0)
    model = BPRMF(4, 6, d=8)
    sampler = TripletSampler(SEQUENCES, 6, seed=0)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(100):
        users, positives, negatives = (torch.as_tensor(x) for x in sampler.sample())
        optimiser.zero_grad()
        model.loss(users, positives, negatives).backward()
        optimiser.step()
    scores = model.predict(torch.arange(4))
    assert torch.isneginf(scores[:, 0]).all()
    assert scores[0, [1, 2, 3]].min() > scores[0, 6]


def test_gru_scores_and_loss():
    model = GRURec(10, max_len=4, d=8).eval()
    seq = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]])
    assert model.predict(seq).shape == (2, 11)
    assert torch.isfinite(model.loss(seq, targets))
