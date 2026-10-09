"""The first real-data check: fp16 as reference, fp16_run1 as noise floor, AWQ as candidate,
for the main and demo probes, each at its own threshold and target FPR.

Run from the repo root, after experiments/phase1/probe/snapshots.py:
    source scripts/dev-env.sh && uv run python experiments/phase1/gate/real_check.py

Writes docs/results/phase1/real_check.json. Report doesn't carry FPR, FNR or AUROC yet, so
those are computed here for context and kept apart from the Report.
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/phase1/probe")]

from common import run_manifest

import hulder
from hulder.gate.check import _auroc

SNAPSHOTS = ROOT / ".cache/snapshots"
PROBES = {"main": ROOT / ".cache/probe/probe_main.npz", "demo": ROOT / ".cache/probe/probe_demo.npz"}
OUT = ROOT / "docs/results/phase1/real_check.json"
SEED = 0


def context(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    flagged = scores > threshold
    return {
        "fpr": float(flagged[labels == 0].mean()),
        "false_positives": int(flagged[labels == 0].sum()),
        "fnr": float((~flagged[labels == 1]).mean()),
        "missed_attacks": int((~flagged[labels == 1]).sum()),
        "auroc": _auroc(scores, labels),
    }


def main() -> None:
    results = {}
    inputs = []
    for name, probe_path in PROBES.items():
        probe = hulder.Probe.load(probe_path)
        paths = {run: SNAPSHOTS / f"{name}_{run}.npz" for run in ("fp16", "fp16_run1", "awq")}
        inputs += [probe_path, *paths.values()]
        reference, rerun, candidate = (hulder.Snapshot.load(p) for p in paths.values())
        # check verifies reference against candidate; the rerun's pairing is checked here.
        assert (rerun.ids == reference.ids).all() and (rerun.labels == reference.labels).all()
        threshold, target_fpr = probe.metadata["threshold"], probe.metadata["target_fpr"]
        report = hulder.check(
            reference.activations,
            candidate.activations,
            reference.labels,
            threshold=threshold,
            target_fpr=target_fpr,
            seed=SEED,
            noise_floor=rerun.activations,
            reference_meta=reference.metadata,
            candidate_meta=candidate.metadata,
            reference_ids=reference.ids,
            candidate_ids=candidate.ids,
            probe=probe,
        )
        labels = reference.labels
        results[name] = {
            "threshold": threshold,
            "target_fpr": target_fpr,
            "n_benign": int((labels == 0).sum()),
            "n_attack": int((labels == 1).sum()),
            "report": {**dataclasses.asdict(report), "verdict": report.verdict.value},
            "exit_code": report.verdict.exit_code,
            "context_not_from_check": {
                run: context(hulder.apply_probe(probe, snap.activations), labels, threshold)
                for run, snap in zip(paths, (reference, rerun, candidate))
            },
        }
        print(f"{name}: {report.verdict.value}")
        print(json.dumps(results[name], indent=2, default=str))
    results["run_manifest"] = run_manifest(seed=SEED, inputs=tuple(p.relative_to(ROOT) for p in inputs))
    OUT.write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
