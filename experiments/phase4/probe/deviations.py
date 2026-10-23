"""Experiment 1b: what a numerical-path deviation does to the Qwen3 probe's scores and decisions.

Each deviated capture is compared with a twin on the majority path that saw the same prompts in
the same batches. Only rows whose activations differ from the twin's are scored. Each is scored
by the frozen probe for its layer and position, and reported as the shift in score and logit and
whether its decision flips at the frozen threshold (docs/results/phase4/full_capture.md):
  whole-capture  the two 2026-10-13 smoke captures on the second path, 64 prompts each
  batch 8        the pilot's canary_check, one batch of 8 prompts
  batch 106      the pilot's capture, one prompt (slabs/train/682)
The first two recorded position -1 only, so they are scored by the two position -1 cells.

Run from the repo root, after train_qwen3.py score:
    source scripts/dev-env.sh && uv run python experiments/phase4/probe/deviations.py

Writes docs/results/phase4/probe_qwen3_deviations.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import train_qwen3 as t

import hulder
from hulder.gate.probe import probe_logits

RUNS = Path(".cache/modal-runs")
OUT = Path("docs/results/phase4/probe_qwen3_deviations.json")
# (name, deviated capture, majority-path twin, {cell: (deviated file, twin file)})
POSITION_LAST = {"18_-1": ("18.npz", "18.npz"), "27_-1": ("27.npz", "27.npz")}
EVERY_CELL = {f"{layer}_{position}": (f"{layer}_{position}.npz",) * 2 for layer, position in t.CELLS}
PAIRS = (
    (
        "whole-capture, 20261013T141533Z rerun",
        "20261013T141533Z/qwen3_4b_bf16_rerun",
        "20261013T141533Z/qwen3_4b_bf16",
        POSITION_LAST,
    ),
    (
        "whole-capture, 20261013T144543Z",
        "20261013T144543Z/qwen3_4b_bf16",
        "20261013T141533Z/qwen3_4b_bf16",
        POSITION_LAST,
    ),
    ("batch 8, pilot canary_check", "20261017T145226Z/canary_check", "20261019T112159Z/u1_canary_check", POSITION_LAST),
    ("batch 106, pilot", "20261017T145226Z/pilot", "20261022T125644Z/kept", EVERY_CELL),
)


def compare(probe: hulder.Probe, *, threshold: float, deviated: hulder.Snapshot, twin: hulder.Snapshot) -> dict:
    """The deviated snapshot's rows that differ from the twin's row for the same prompt id, each
    scored under both, with the shift and whether the decision flips."""
    row_of = {str(i): r for r, i in enumerate(twin.ids)}
    ids = [str(i) for i in deviated.ids]
    mine = np.asarray(deviated.activations)
    theirs = np.asarray(twin.activations)[[row_of[i] for i in ids]]
    differ = np.flatnonzero((mine != theirs).any(axis=1))
    out_rows = []
    if len(differ):
        s_dev, s_twin = hulder.apply_probe(probe, mine[differ]), hulder.apply_probe(probe, theirs[differ])
        l_dev, l_twin = probe_logits(probe, mine[differ]), probe_logits(probe, theirs[differ])
        for k, r in enumerate(differ):
            out_rows.append(
                {
                    "id": ids[r],
                    "label": int(deviated.labels[r]),
                    "twin_score": float(s_twin[k]),
                    "score": float(s_dev[k]),
                    "delta_score": float(s_dev[k] - s_twin[k]),
                    "delta_logit": float(l_dev[k] - l_twin[k]),
                    "flip": bool((s_dev[k] > threshold) != (s_twin[k] > threshold)),
                }
            )
    shifts = np.abs([r["delta_score"] for r in out_rows]) if out_rows else np.zeros(0)
    logits = np.abs([r["delta_logit"] for r in out_rows]) if out_rows else np.zeros(0)
    return {
        "compared": len(ids),
        "deviated": len(out_rows),
        "flips": sum(r["flip"] for r in out_rows),
        "mean_abs_delta_score": float(shifts.mean()) if len(shifts) else None,
        "max_abs_delta_score": float(shifts.max()) if len(shifts) else None,
        "mean_abs_delta_logit": float(logits.mean()) if len(logits) else None,
        "max_abs_delta_logit": float(logits.max()) if len(logits) else None,
        "rows": out_rows,
    }


def main() -> None:
    selection = json.loads(t.SELECTION.read_text())
    if hashlib.sha256(t.SELECTION.read_bytes()).hexdigest() != t.SELECTION_SHA256:
        raise SystemExit(f"{t.SELECTION} isn't the committed selection")
    result = {"selection_sha256": t.SELECTION_SHA256, "pairs": []}
    for name, deviated, twin, cells in PAIRS:
        entry = {"name": name, "deviated": deviated, "twin": twin, "cells": {}}
        for cell, (dev_file, twin_file) in cells.items():
            probe = hulder.Probe.load(t.PROBES / f"{cell}.npz")
            threshold = selection["cells"][cell]["threshold"]
            entry["cells"][cell] = {
                "threshold": threshold,
                **compare(
                    probe,
                    threshold=threshold,
                    deviated=hulder.Snapshot.load(RUNS / deviated / dev_file),
                    twin=hulder.Snapshot.load(RUNS / twin / twin_file),
                ),
            }
            c = entry["cells"][cell]
            print(
                f"{name}, {cell}: {c['deviated']} of {c['compared']} rows deviated, {c['flips']} flips, "
                f"max |dscore| {c['max_abs_delta_score']}, max |dlogit| {c['max_abs_delta_logit']}"
            )
        result["pairs"].append(entry)
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
