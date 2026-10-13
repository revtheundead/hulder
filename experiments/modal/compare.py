"""Summarise a Modal smoke run: per config the GPU, KV-cache and kernel log lines, timings and
estimated cost, and per layer how far the bf16 rerun and FP8 moved activations from bf16. No
verdict: 64 prompts and no probe for Qwen3 are too few to judge one.

    uv run python experiments/modal/compare.py .cache/modal-runs/<run id>

Writes comparison.json next to the run's manifest.json and prints it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import smoke_lib as lib

import hulder

REFERENCE = "qwen3_4b_bf16"
AGAINST = ("qwen3_4b_bf16_rerun", "qwen3_4b_fp8")


def summarise(run_dir: Path) -> dict:
    run = json.loads((run_dir / "manifest.json").read_text())
    configs = {}
    for name, entry in run["configs"].items():
        path = run_dir / name / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        worker = manifest.get("worker", {})
        configs[name] = {
            **entry,
            "nvidia_smi": manifest.get("nvidia_smi"),
            "capture_seconds": worker.get("capture_seconds"),
            "last_tokens": worker.get("last_tokens"),
            **manifest.get("log", {}),
        }
    shifts = {}
    for other in AGAINST:
        if not all((run_dir / n / f"{lib.LAYERS[0]}.npz").exists() for n in (REFERENCE, other)):
            continue
        for layer in lib.LAYERS:
            ref = hulder.Snapshot.load(run_dir / REFERENCE / f"{layer}.npz")
            cand = hulder.Snapshot.load(run_dir / other / f"{layer}.npz")
            if not np.array_equal(ref.ids, cand.ids):
                raise SystemExit(f"{other} layer {layer} holds other prompts than {REFERENCE}")
            rel = lib.relative_l2(ref.activations, cand.activations)
            shifts[f"{other} vs {REFERENCE}, layer {layer}"] = {
                "relative_l2_mean": float(rel.mean()),
                "relative_l2_max": float(rel.max()),
                "bit_identical": f"{int((rel == 0).sum())}/{len(rel)}",
            }
    keep = ("run_id", "commit", "estimated_usd", "stopped")
    return {**{k: run.get(k) for k in keep}, "configs": configs, "shifts": shifts}


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    summary = summarise(run_dir)
    (run_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
