"""Snapshot files for the real-data checks: each check set under fp16, fp16_run1 and AWQ.

Check sets: "main" is the S-Labs calibration split (the main probe's threshold set), "demo"
the boundary-pairs rows demo_splits.json marks calibration (the demo probe's). All three
captures ran every prompt of splits.parquet in the same order and batches (extract.py), so
these rows saw the same batch-mates in each run.

Run from the repo root, after extract.py for all three configs:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/snapshots.py

Writes .cache/snapshots/{main,demo}_{fp16,fp16_run1,awq}.npz: layer 20, position -1, fp16.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import SPLITS, Activations

import hulder
from hulder.capture.base_model import base_model_for

OUT = Path(".cache/snapshots")
RUNS = ("fp16", "fp16_run1", "awq")
LAYER, POSITION = 20, -1


def check_sets() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(SPLITS)
    demo = json.loads(Path(".cache/probe/demo_splits.json").read_text())
    return {
        "main": df[df["split"] == "calibration"],
        "demo": df[df["id"].map(demo) == "calibration"],
    }


def metadata(acts: Activations) -> dict:
    """Only keys comparable across runs: the run's own provenance stays in its manifest."""
    manifest = acts.capture_manifest
    cfg = manifest["capture_config"]
    base_model, warning = base_model_for(cfg["model"], cfg["revision"], cfg["quantization"])
    if warning:
        raise SystemExit(warning)
    return {
        "base_model": base_model,
        "model": cfg["model"],
        "revision": cfg["revision"],
        "quantization": cfg["quantization"],
        "dtype": cfg["dtype"],
        "batch_size": cfg["batch_size"],
        # fp16_run1 predates run_manifest and records the version at the top level.
        "vllm_version": manifest.get("packages", {}).get("vllm") or manifest["vllm"],
        "system_prompt": cfg["system_prompt"],
        "layer": LAYER,
        "position": POSITION,
        "splits_sha256": manifest.get("inputs_sha256", {}).get(str(SPLITS)) or manifest["splits_sha256"],
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sets = check_sets()
    for run in RUNS:
        acts = Activations.load(Path(".cache/probe") / run)
        meta = metadata(acts)
        for name, rows in sets.items():
            path = OUT / f"{name}_{run}.npz"
            hulder.Snapshot(
                ids=rows["id"].to_numpy(dtype=str),
                labels=rows["label"].to_numpy(),
                activations=acts.at(rows["id"], LAYER, POSITION).astype(np.float16),
                metadata=meta,
            ).save(path)
            print(f"{path}: {len(rows)} examples ({int((rows['label'] == 0).sum())} benign)")


if __name__ == "__main__":
    main()
