"""Train the quickstart's demo probe on boundary-pairs (the dataset decision rule triggered, probe_main.md).

DEMO ONLY. Small data and a 5% FPR target; never use it as a real prompt-attack monitor.

Reuses the main probe's layer, position and C, so boundary-pairs has no validation split.
Split by scenario (50 scenarios x 12 pairs), which keeps every attack with its paired benign
request and keeps a scenario's shared context out of the other splits:
  test         the dataset's own test split, 10 scenarios, 240 examples
  train        25 of the remaining 40 scenarios, 600 examples (seed 0)
  calibration  the other 15, 360 examples; sets the 5% FPR threshold

Run from the repo root, after train.py:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/train_demo.py

Writes .cache/probe/probe_demo.npz, .cache/probe/demo_splits.json and
docs/results/phase1/probe_demo.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score
from train import at_threshold, auroc_ci, fit, threshold_for_fpr

import hulder

BPAIRS = ("3nesdeniz/agentic-prompt-injection-boundary-pairs", "a5682e7573e1c7bc4b12e64d49c0dcd90ca776cf")
ACTS = Path(".cache/probe/fp16")
MAIN = Path(".cache/probe/probe_main.npz")
OUT_PROBE = Path(".cache/probe/probe_demo.npz")
OUT_SPLITS = Path(".cache/probe/demo_splits.json")
OUT_REPORT = Path("docs/results/phase1/probe_demo.json")
TARGET_FPR = 0.05
N_TRAIN_SCENARIOS = 25
SEED = 0
ATTRIBUTION = (
    "Contains data derived from 3nesdeniz/agentic-prompt-injection-boundary-pairs "
    "(Deniz 2026, doi:10.5281/zenodo.21378693), CC-BY-4.0."
)


def main() -> None:
    bp = pd.concat(
        pd.read_parquet(hf_hub_download(BPAIRS[0], f"data/{s}.parquet", repo_type="dataset", revision=BPAIRS[1]))
        for s in ["train", "validation", "test"]
    )
    bp["id"] = "bpairs/" + bp["id"]

    rest = sorted(bp.loc[bp["split"] != "test", "scenario_id"].unique())
    rng = np.random.default_rng(SEED)
    train_scn = set(rng.choice(rest, size=N_TRAIN_SCENARIOS, replace=False))
    bp["demo_split"] = np.where(
        bp["split"] == "test", "test", np.where(bp["scenario_id"].isin(train_scn), "train", "calibration")
    )
    assert (bp.groupby("pair_id")["demo_split"].nunique() == 1).all()
    assert (bp.groupby("scenario_id")["demo_split"].nunique() == 1).all()

    main_probe = hulder.Probe.load(MAIN)
    manifest = json.loads((ACTS / "manifest.json").read_text())
    cap = manifest["capture_config"]
    li = cap["layers"].index(main_probe.metadata["layer"])
    pi = cap["positions"].index(main_probe.metadata["position"])
    row_of = {i: r for r, i in enumerate(json.loads((ACTS / "ids.json").read_text()))}
    acts = np.load(ACTS / "activations.npy", mmap_mode="r")

    def part(s: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sub = bp[bp["demo_split"] == s]
        x = np.asarray(acts[[row_of[i] for i in sub["id"]], li, pi])
        return x, sub["label"].to_numpy(), sub["pair_id"].to_numpy()

    meta = {
        **{k: main_probe.metadata[k] for k in ("model", "revision", "layer", "position", "system_prompt", "dtype", "C")},
        "task": "prompt attack (injection or jailbreak)",
        "demo_only": True,
        "note": "Quickstart demo probe. Trained on 600 examples with a 5% FPR target. Not a real monitor.",
        "training_data": "boundary-pairs@a5682e75, 25 scenarios (demo_splits.json)",
        "attribution": ATTRIBUTION,
    }
    x_tr, y_tr, _ = part("train")
    probe = fit(x_tr, y_tr, main_probe.metadata["C"], meta)
    x_ca, y_ca, _ = part("calibration")
    x_te, y_te, pairs_te = part("test")
    s_ca, s_te = hulder.apply_probe(probe, x_ca), hulder.apply_probe(probe, x_te)
    t = threshold_for_fpr(s_ca[y_ca == 0], TARGET_FPR)
    probe = hulder.Probe(probe.weights, probe.bias, probe.mean, probe.std, {**meta, "threshold": t, "target_fpr": TARGET_FPR})

    sizes = {s: {"n": int((bp["demo_split"] == s).sum()), "scenarios": int(bp.loc[bp["demo_split"] == s, "scenario_id"].nunique())} for s in ("train", "calibration", "test")}
    report = {
        "split_sizes": sizes,
        "layer": meta["layer"],
        "position": meta["position"],
        "C": meta["C"],
        "threshold": t,
        "target_fpr": TARGET_FPR,
        "calibration": {"auroc": roc_auc_score(y_ca, s_ca), **at_threshold(s_ca, y_ca, t)},
        "test": {**auroc_ci(s_te, y_te, clusters=pairs_te), **at_threshold(s_te, y_te, t), "bootstrap": "pair-clustered"},
        "attribution": ATTRIBUTION,
    }
    probe.save(OUT_PROBE)
    OUT_SPLITS.write_text(json.dumps(dict(zip(bp["id"], bp["demo_split"])), indent=0) + "\n")
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
