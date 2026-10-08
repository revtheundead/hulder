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

import dataclasses
import json
from pathlib import Path

import numpy as np
from common import ACTS, BPAIRS, BPAIRS_ATTRIBUTION, Activations, load_bpairs, run_manifest
from sklearn.metrics import roc_auc_score
from train import at_threshold, auroc_ci, fit, threshold_for_fpr

import hulder

MAIN = Path(".cache/probe/probe_main.npz")
OUT_PROBE = Path(".cache/probe/probe_demo.npz")
OUT_SPLITS = Path(".cache/probe/demo_splits.json")
OUT_REPORT = Path("docs/results/phase1/probe_demo.json")
TARGET_FPR = 0.05
N_TRAIN_SCENARIOS = 25
SEED = 0
# Copied from the main probe, so the demo needs no validation split (the dataset decision Outcome).
INHERITED = ("model", "revision", "layer", "position", "system_prompt", "dtype", "C")


def main() -> None:
    bp = load_bpairs()

    rest = sorted(bp.loc[bp["split"] != "test", "scenario_id"].unique())
    rng = np.random.default_rng(SEED)
    train_scn = set(rng.choice(rest, size=N_TRAIN_SCENARIOS, replace=False))
    bp["demo_split"] = np.where(
        bp["split"] == "test", "test", np.where(bp["scenario_id"].isin(train_scn), "train", "calibration")
    )
    assert (bp.groupby("pair_id")["demo_split"].nunique() == 1).all()
    assert (bp.groupby("scenario_id")["demo_split"].nunique() == 1).all()

    main_probe = hulder.Probe.load(MAIN)
    acts = Activations.load(ACTS)

    def split_arrays(s: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Activations, labels and pair ids of one demo split, at the main probe's layer and position."""
        sub = bp[bp["demo_split"] == s]
        return acts.for_probe(sub["id"], main_probe), sub["label"].to_numpy(), sub["pair_id"].to_numpy()

    meta = {
        **{k: main_probe.metadata[k] for k in INHERITED},
        "task": "prompt attack (injection or jailbreak)",
        "demo_only": True,
        "note": "Quickstart demo probe. Trained on 600 examples with a 5% FPR target. Not a real monitor.",
        "training_data": f"{BPAIRS.repo}@{BPAIRS.revision[:8]}, 25 scenarios (demo_splits.json)",
        "attribution": BPAIRS_ATTRIBUTION,
    }
    x_tr, y_tr, _ = split_arrays("train")
    probe = fit(x_tr, y_tr, main_probe.metadata["C"], meta)
    x_ca, y_ca, _ = split_arrays("calibration")
    x_te, y_te, pairs_te = split_arrays("test")
    s_ca, s_te = hulder.apply_probe(probe, x_ca), hulder.apply_probe(probe, x_te)
    t = threshold_for_fpr(s_ca[y_ca == 0], TARGET_FPR)
    probe = dataclasses.replace(probe, metadata={**meta, "threshold": t, "target_fpr": TARGET_FPR})

    sizes = {
        s: {
            "n": int((bp["demo_split"] == s).sum()),
            "scenarios": int(bp.loc[bp["demo_split"] == s, "scenario_id"].nunique()),
        }
        for s in ("train", "calibration", "test")
    }
    report = {
        "demo_only": True,
        "note": meta["note"],
        "split_sizes": sizes,
        "layer": meta["layer"],
        "position": meta["position"],
        "C": meta["C"],
        "threshold": t,
        "target_fpr": TARGET_FPR,
        "calibration": {"auroc": roc_auc_score(y_ca, s_ca), **at_threshold(s_ca, y_ca, t)},
        "test": {
            **auroc_ci(s_te, y_te, clusters=pairs_te),
            **at_threshold(s_te, y_te, t),
            "bootstrap": "pair-clustered",
        },
        "attribution": BPAIRS_ATTRIBUTION,
        "run_manifest": run_manifest(seed=SEED, inputs=(MAIN, ACTS / "activations.npy")),
    }
    probe.save(OUT_PROBE)
    OUT_SPLITS.write_text(json.dumps(dict(zip(bp["id"], bp["demo_split"])), indent=0) + "\n")
    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
