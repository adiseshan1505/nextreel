"""Full-ranking HR@K, NDCG@K, coverage and popularity bias."""

import numpy as np
import torch


def hit_rate(ranks, k=10):
    return float(np.mean(np.asarray(ranks) <= k))


def ndcg(ranks, k=10):
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)))


def mask_history(scores, histories, targets=None):
    scores = scores.clone()
    kept = None if targets is None else scores.gather(1, targets[:, None])
    rows = np.repeat(np.arange(len(histories)), [len(h) for h in histories])
    cols = np.concatenate([np.asarray(h, dtype=np.int64) for h in histories])
    scores[torch.as_tensor(rows), torch.as_tensor(cols)] = float("-inf")
    scores[:, 0] = float("-inf")
    if targets is not None:
        scores.scatter_(1, targets[:, None], kept)
    return scores


def target_ranks(scores, targets, generator=None):
    target_scores = scores.gather(1, targets[:, None])
    higher = (scores > target_scores).sum(1)
    ties = (scores == target_scores).sum(1) - 1
    offset = (torch.rand(len(targets), generator=generator) * (ties + 1)).floor().long()
    return (higher + 1 + offset).numpy()


def sample_negatives(histories, targets, n_items, n_negatives, rng):
    negatives = np.empty((len(targets), n_negatives), dtype=np.int64)
    for i, (history, target) in enumerate(zip(histories, targets)):
        seen = np.zeros(n_items + 1, dtype=bool)
        seen[0] = True
        seen[np.asarray(history, dtype=np.int64)] = True
        seen[target] = True
        negatives[i] = rng.choice(np.flatnonzero(~seen), n_negatives, replace=False)
    return negatives


def sampled_ranks(scores, targets, negatives, generator=None):
    candidates = torch.cat([targets[:, None], negatives], dim=1)
    first = torch.zeros(len(targets), dtype=torch.long)
    return target_ranks(scores.gather(1, candidates), first, generator)


def coverage(recommended, n_items):
    return len(np.unique(recommended)) / n_items


def popularity_percentile(item_counts):
    counts = np.asarray(item_counts[1:])
    percentile = np.searchsorted(np.sort(counts), counts, side="left") / len(counts)
    return np.concatenate([[0.0], percentile])


def popularity_bias(recommended, percentile):
    return float(np.mean(percentile[recommended]))


class Evaluator:
    def __init__(self, histories, targets, item_counts, k=10, n_negatives=100, seed=0):
        self.histories = histories
        self.targets = np.asarray(targets, dtype=np.int64)
        self.n_items = len(item_counts) - 1
        self.k = k
        self.seed = seed
        self.percentile = popularity_percentile(item_counts)
        self.negatives = None
        if n_negatives:
            rng = np.random.default_rng(seed)
            self.negatives = sample_negatives(histories, self.targets, self.n_items, n_negatives, rng)

    def evaluate(self, score_fn, batch_size=256, sampled=True):
        generator = torch.Generator().manual_seed(self.seed)
        sampled = sampled and self.negatives is not None
        ranks, sampled_rank_list, recommended = [], [], []
        for start in range(0, len(self.targets), batch_size):
            users = np.arange(start, min(start + batch_size, len(self.targets)))
            scores = torch.as_tensor(score_fn(users), dtype=torch.float32)
            targets = torch.as_tensor(self.targets[users])
            if sampled:
                negatives = torch.as_tensor(self.negatives[users])
                sampled_rank_list.append(sampled_ranks(scores, targets, negatives, generator))
            masked = mask_history(scores, [self.histories[u] for u in users], targets)
            ranks.append(target_ranks(masked, targets, generator))
            recommended.append(masked.topk(self.k, dim=1).indices.numpy())
        ranks = np.concatenate(ranks)
        recommended = np.concatenate(recommended)
        k = self.k
        metrics = {
            f"hr@{k}": hit_rate(ranks, k),
            f"ndcg@{k}": ndcg(ranks, k),
            f"coverage@{k}": coverage(recommended, self.n_items),
            f"popularity@{k}": popularity_bias(recommended, self.percentile),
        }
        if sampled:
            sampled_rank_array = np.concatenate(sampled_rank_list)
            metrics[f"sampled_hr@{k}"] = hit_rate(sampled_rank_array, k)
            metrics[f"sampled_ndcg@{k}"] = ndcg(sampled_rank_array, k)
        return metrics
