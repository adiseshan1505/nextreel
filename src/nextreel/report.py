import argparse
import json
from pathlib import Path

import numpy as np

ORDER = ["popularity", "item2item", "mf_bpr", "gru", "sasrec", "sasrec_no_position"]
LABELS = {
    "popularity": "Most popular",
    "item2item": "Item-to-item",
    "mf_bpr": "Matrix factorisation (BPR)",
    "gru": "GRU",
    "sasrec": "NextReel (SASRec)",
    "sasrec_no_position": "NextReel, no position vectors",
}


def collect(runs_dir):
    results = {}
    for path in sorted(Path(runs_dir).glob("*/seed*/metrics.json")):
        result = json.loads(path.read_text())
        results.setdefault(result["name"], []).append(result)
    return results


def ordered(results):
    known = [name for name in ORDER if name in results]
    return known + sorted(set(results) - set(known))


def mean_std(values):
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):.4f} ± {std:.4f}"


def metric_row(runs, metrics):
    return [mean_std([run["test"][m] for run in runs]) for m in metrics]


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render(results, k=10):
    metrics = [f"hr@{k}", f"ndcg@{k}", f"coverage@{k}", f"popularity@{k}"]
    header = ["Model", f"HR@{k}", f"NDCG@{k}", f"Coverage@{k}", f"Popularity@{k}", "Seeds"]
    rows = [
        [LABELS.get(name, name), *metric_row(results[name], metrics), str(len(results[name]))]
        for name in ordered(results)
    ]
    return "\n".join([
        "# Results",
        "",
        "Full-catalogue test metrics, mean ± std over seeds.",
        "",
        markdown_table(header, rows),
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect run metrics into a markdown table.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="results/results.md")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    results = collect(args.runs_dir)
    if not results:
        raise SystemExit(f"no metrics.json found under {args.runs_dir}")
    Path(args.out).write_text(render(results, args.k))
    print(f"wrote {args.out} with {len(results)} models")


if __name__ == "__main__":
    main()
