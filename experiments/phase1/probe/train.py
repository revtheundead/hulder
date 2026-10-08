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

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import ACTS, SLABS, SPLITS, Activations, load_bpairs, run_manifest
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import hulder

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
    if n == 0:
        return {"rate": None, "k": 0, "n": 0, "ci95": None}
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
    acts = Activations.load(ACTS)
    assert list(acts.row_of) == df["id"].tolist()
    cap = acts.capture_manifest["capture_config"]
    split = df["split"].to_numpy()
    mask_of = {s: split == s for s in np.unique(split)}
    mask_of["bpairs"] = df["split"].str.startswith("bpairs_").to_numpy()
    ids_of = {s: df["id"].to_numpy()[m] for s, m in mask_of.items()}
    labels_of = {s: df["label"].to_numpy()[m] for s, m in mask_of.items()}

    # Layer x C grid on validation, both positions reported, selection at position -1.
    grid = []
    for position in cap["positions"]:
        for layer in cap["layers"]:
            x_train, x_val = (acts.at(ids_of[s], layer, position) for s in ("train", "validation"))
            for c in C_GRID:
                p = fit(x_train, labels_of["train"], c, {})
                val = roc_auc_score(labels_of["validation"], hulder.apply_probe(p, x_val))
                grid.append({"position": position, "layer": layer, "C": c, "val_auroc": val})
                print(f"pos {position:>2} layer {layer:>2} C {c:<6} val AUROC {val:.4f}", flush=True)
    best = max((g for g in grid if g["position"] == -1), key=lambda g: g["val_auroc"])

    meta = {
        "model": cap["model"],
        "revision": cap["revision"],
        "layer": best["layer"],
        "position": best["position"],
        "system_prompt": cap["system_prompt"],
        "dtype": cap["dtype"],
        "task": "prompt attack (injection or jailbreak)",
        "training_data": f"{SLABS.repo}@{SLABS.revision[:8]}, train split after dedup (probe_splits.json)",
        "C": best["C"],
    }
    probe = fit(acts.at(ids_of["train"], best["layer"], best["position"]), labels_of["train"], best["C"], meta)
    score = {s: hulder.apply_probe(probe, acts.for_probe(ids_of[s], probe)) for s in ["calibration", "test", "bpairs"]}
    t = threshold_for_fpr(score["calibration"][labels_of["calibration"] == 0], TARGET_FPR)
    probe = dataclasses.replace(probe, metadata={**meta, "threshold": t, "target_fpr": TARGET_FPR})

    pair_of = dict(zip(*load_bpairs()[["id", "pair_id"]].to_numpy().T))
    bp = auroc_ci(score["bpairs"], labels_of["bpairs"], clusters=np.array([pair_of[i] for i in ids_of["bpairs"]]))
    report = {
        "selected": best,
        "grid": grid,
        "threshold": t,
        "calibration": at_threshold(score["calibration"], labels_of["calibration"], t),
        "test": {**auroc_ci(score["test"], labels_of["test"]), **at_threshold(score["test"], labels_of["test"], t)},
        "bpairs_all_1200": {
            **bp,
            **at_threshold(score["bpairs"], labels_of["bpairs"], t),
            "bootstrap": "pair-clustered",
        },
        "adr_0004_rule": {
            "threshold_auroc": AUROC_RULE,
            "point_estimate": bp["auroc"],
            "outcome": "keep S-Labs probe for the quickstart"
            if bp["auroc"] >= AUROC_RULE
            else "train a demo probe on boundary-pairs",
        },
        "capture_manifest": acts.capture_manifest,
        "run_manifest": run_manifest(seed=SEED, inputs=(SPLITS, ACTS / "activations.npy")),
    }
    OUT_PROBE.parent.mkdir(parents=True, exist_ok=True)
    probe.save(OUT_PROBE)
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("grid", "capture_manifest", "run_manifest")}, indent=2)
    )


if __name__ == "__main__":
    main()
