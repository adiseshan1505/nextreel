import pytest
import torch

from nextreel.models import SASRec


def make_sasrec(**kwargs):
    torch.manual_seed(0)
    return SASRec(n_items=30, max_len=8, d=16, n_layers=2, n_heads=2, dropout=0.0, **kwargs).eval()


def batch():
    seq = torch.tensor([[0, 0, 0, 4, 9, 2, 7, 1], [3, 5, 6, 8, 10, 11, 12, 13]])
    targets = torch.tensor([[0, 0, 0, 9, 2, 7, 1, 5], [5, 6, 8, 10, 11, 12, 13, 14]])
    return seq, targets


def test_scores_cover_every_item_and_mask_padding():
    scores = make_sasrec().predict(batch()[0])
    assert scores.shape == (2, 31)
    assert torch.isinf(scores[:, 0]).all()
    assert torch.isfinite(scores[:, 1:]).all()


def test_future_items_do_not_change_earlier_outputs():
    model = make_sasrec()
    seq = batch()[0]
    changed = seq.clone()
    changed[:, -1] = 20
    with torch.no_grad():
        before, after = model.encode(seq), model.encode(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_attention_never_looks_ahead_or_at_padding():
    with torch.no_grad():
        _, attentions = make_sasrec().encode(batch()[0], return_attention=True)
    weights = attentions[-1]
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 8))
    assert weights.triu(1).abs().max() == 0
    assert weights[0, :, -1, :3].abs().max() == 0


def test_loss_is_finite_and_trains_item_embeddings():
    model = make_sasrec().train()
    loss = model.loss(*batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(30.0)).item(), abs=0.5)
    assert model.item_emb.weight.grad[1:].abs().sum() > 0


def test_order_only_matters_through_positions_and_masking():
    seq = batch()[0][1:]
    shuffled = seq[:, torch.tensor([1, 0, 2, 3, 4, 5, 6, 7])]
    with_pos = make_sasrec()
    assert not torch.allclose(with_pos.predict(seq), with_pos.predict(shuffled))
    without_pos = make_sasrec(use_position=False)
    assert torch.isfinite(without_pos.predict(seq)[:, 1:]).all()
