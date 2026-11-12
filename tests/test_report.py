import json

from nextreel.report import collect, main, mean_std, render


def write_run(runs_dir, name, seed, hr):
    out = runs_dir / name / f"seed{seed}"
    out.mkdir(parents=True)
    test = {
        "hr@10": hr, "ndcg@10": hr / 2, "coverage@10": 0.3, "popularity@10": 0.9,
        "sampled_hr@10": hr * 3, "sampled_ndcg@10": hr,
    }
    (out / "metrics.json").write_text(json.dumps({"name": name, "seed": seed, "test": test}))


def test_mean_std_formatting():
    assert mean_std([0.1, 0.3]) == "0.2000 ± 0.1414"
    assert mean_std([0.5]) == "0.5000 ± 0.0000"


def test_report_orders_baselines_before_nextreel(tmp_path):
    for seed, hr in enumerate([0.2, 0.22, 0.24]):
        write_run(tmp_path, "sasrec", seed, hr)
    write_run(tmp_path, "popularity", 0, 0.05)
    results = collect(tmp_path)
    assert len(results["sasrec"]) == 3
    text = render(results)
    assert text.index("Most popular") < text.index("NextReel (SASRec)")
    assert "0.2200 ± 0.0200" in text
    assert "HR@10 sampled" in text


def test_report_writes_markdown(tmp_path):
    write_run(tmp_path / "runs", "mf_bpr", 0, 0.1)
    out = tmp_path / "results.md"
    main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out)])
    assert out.read_text().startswith("# Results")
