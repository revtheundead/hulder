"""Train the v0.1 example probe and apply the dataset decision's boundary-pairs rule.

Split roles (README §8):
  train        fits weights and the standardisation mean/std
  validation   picks layer and C, at position -1 (README §6 D2 default)
  calibration  sets the threshold for 1% FPR on benign scores
  test         in-distribution metrics, touched once
  bpairs_*     held out; scored only, for the AUROC >= 0.85 rule

The threshold is the empirical (1 - 0.01) quantile of benign calibration scores, rounded
up. This is provisional: conformal vs binomial-tail is the Phase 2 written decision (README §7).

Run from the repo root, after extract.py --config fp16:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/train.py

Writes .cache/probe/probe_main.npz and docs/results/phase1/probe_main.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import hulder

ACTS = Path(".cache/probe/fp16")
SPLITS = Path(".cache/probe/splits.parquet")
OUT_PROBE = Path(".cache/probe/probe_main.npz")
OUT_REPORT = Path("docs/results/phase1/probe_main.json")
TARGET_FPR = 0.01
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
AUROC_RULE = 0.85  # The dataset decision
N_BOOT = 2000
SEED = 0


def fit(x: np.ndarray, y: np.ndarray, c: float, meta: dict) -> hulder.Probe:
    x = x.astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=c, max_iter=5000).fit((x - mean) / std, y)
    return hulder.Probe(weights=clf.coef_[0], bias=float(clf.intercept_[0]), mean=mean, std=std, metadata=meta)


def threshold_for_fpr(benign_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(benign_scores, 1 - fpr, method="higher"))


def rate_ci(k: int, n: int) -> dict:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return {"rate": k / n, "k": k, "n": n, "ci95": [ci.low, ci.high]}


def at_threshold(scores: np.ndarray, y: np.ndarray, t: float) -> dict:
    flagged = scores > t
    return {
        "fpr": rate_ci(int((flagged & (y == 0)).sum()), int((y == 0).sum())),
        "fnr": rate_ci(int((~flagged & (y == 1)).sum()), int((y == 1).sum())),
    }


def auroc_ci(scores: np.ndarray, y: np.ndarray, clusters: np.ndarray | None = None) -> dict:
    """Percentile bootstrap. With clusters, resamples whole clusters (boundary-pairs pairs)."""
    rng = np.random.default_rng(SEED)
    groups = np.arange(len(y)) if clusters is None else clusters
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], scores[idx]))
    return {"auroc": roc_auc_score(y, scores), "ci95": np.percentile(boots, [2.5, 97.5]).tolist()}


def main() -> None:
    df = pd.read_parquet(SPLITS)
    acts = np.load(ACTS / "activations.npy", mmap_mode="r")
    manifest = json.loads((ACTS / "manifest.json").read_text())
    assert json.loads((ACTS / "ids.json").read_text()) == df["id"].tolist()
    cap = manifest["capture_config"]
    y = df["label"].to_numpy()
    rows = {s: np.flatnonzero(df["split"].to_numpy() == s) for s in df["split"].unique()}
    rows["bpairs"] = np.flatnonzero(df["split"].str.startswith("bpairs_").to_numpy())

    def xs(split: str, li: int, pi: int) -> np.ndarray:
        return np.asarray(acts[rows[split], li, pi])

    # Layer x C grid on validation, both positions reported, selection at position -1.
    grid = []
    for pi, pos in enumerate(cap["positions"]):
        for li, layer in enumerate(cap["layers"]):
            for c in C_GRID:
                p = fit(xs("train", li, pi), y[rows["train"]], c, {})
                val = roc_auc_score(y[rows["validation"]], hulder.apply_probe(p, xs("validation", li, pi)))
                grid.append({"position": pos, "layer": layer, "C": c, "val_auroc": val})
                print(f"pos {pos:>2} layer {layer:>2} C {c:<6} val AUROC {val:.4f}", flush=True)
    best = max((g for g in grid if g["position"] == -1), key=lambda g: g["val_auroc"])
    li, pi = cap["layers"].index(best["layer"]), cap["positions"].index(best["position"])

    meta = {
        "model": cap["model"],
        "revision": cap["revision"],
        "layer": best["layer"],
        "position": best["position"],
        "system_prompt": cap["system_prompt"],
        "dtype": cap["dtype"],
        "task": "prompt attack (injection or jailbreak)",
        "training_data": "S-Labs/prompt-injection-dataset@002a9dd1, train split after dedup (probe_splits.json)",
        "C": best["C"],
    }
    probe = fit(xs("train", li, pi), y[rows["train"]], best["C"], meta)
    score = {s: hulder.apply_probe(probe, xs(s, li, pi)) for s in ["calibration", "test", "bpairs"]}
    t = threshold_for_fpr(score["calibration"][y[rows["calibration"]] == 0], TARGET_FPR)
    probe = hulder.Probe(probe.weights, probe.bias, probe.mean, probe.std, {**meta, "threshold": t, "target_fpr": TARGET_FPR})

    bp_pairs = df["id"].to_numpy()[rows["bpairs"]]
    bp_pairs = np.array([i.rsplit("_", 1)[0] for i in bp_pairs])
    bp = auroc_ci(score["bpairs"], y[rows["bpairs"]], clusters=bp_pairs)
    report = {
        "selected": best,
        "grid": grid,
        "threshold": t,
        "calibration": at_threshold(score["calibration"], y[rows["calibration"]], t),
        "test": {**auroc_ci(score["test"], y[rows["test"]]), **at_threshold(score["test"], y[rows["test"]], t)},
        "bpairs_all_1200": {
            **bp,
            **at_threshold(score["bpairs"], y[rows["bpairs"]], t),
            "bootstrap": "pair-clustered",
        },
        "adr_0004_rule": {
            "threshold_auroc": AUROC_RULE,
            "point_estimate": bp["auroc"],
            "outcome": "keep S-Labs probe for the quickstart" if bp["auroc"] >= AUROC_RULE else "train a demo probe on boundary-pairs",
        },
        "capture_manifest": manifest,
    }
    OUT_PROBE.parent.mkdir(parents=True, exist_ok=True)
    probe.save(OUT_PROBE)
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("grid", "capture_manifest")}, indent=2))


if __name__ == "__main__":
    main()
