"""Summarise a Modal capture run: per run the GPU, KV-cache and kernel log lines, timings and
whether it passed, and per layer how far each run moved activations from the reference, the
run's first. No verdict: 64 prompts and no probe for Qwen3 are too few to judge one.

    uv run python experiments/modal/compare.py .cache/modal-runs/<run id>

Writes comparison.json next to the run's manifest.json and prints it. Leaves out vLLM's second
"Loading weights took" line, which can exceed the whole worker's run time and so isn't a duration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import smoke_lib as lib

import hulder


def summarise(run_dir: Path) -> dict:
    run = json.loads((run_dir / "manifest.json").read_text())
    labels = list(run["configs"])
    runs = {}
    for label in labels:
        path = run_dir / label / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        worker = manifest.get("worker", {})
        log = manifest.get("log", {})
        runs[label] = {
            **run["configs"][label],
            "nvidia_smi": manifest.get("nvidia_smi"),
            "capture_seconds": worker.get("capture_seconds"),
            "last_tokens": worker.get("last_tokens"),
            "kv_cache": log.get("kv_cache", []),
            "kernels": log.get("kernels", []),
            "timings": [line for line in log.get("timings", []) if "Loading weights took" not in line],
        }
    saved = [label for label in labels if (run_dir / label / f"{lib.LAYERS[0]}.npz").exists()]
    shifts = {}
    reference = saved[0] if saved else None
    for other in saved[1:]:
        for layer in lib.LAYERS:
            ref = hulder.Snapshot.load(run_dir / reference / f"{layer}.npz")
            cand = hulder.Snapshot.load(run_dir / other / f"{layer}.npz")
            if not np.array_equal(ref.ids, cand.ids):
                raise SystemExit(f"{other} layer {layer} holds other prompts than {reference}")
            rel = lib.relative_l2(ref.activations, cand.activations)
            keys = ref.metadata.keys() | cand.metadata.keys()
            shifts[f"{other} vs {reference}, layer {layer}"] = {
                "relative_l2_mean": float(rel.mean()),
                "relative_l2_max": float(rel.max()),
                "bit_identical": f"{int((rel == 0).sum())}/{len(rel)}",
                "metadata_differs": sorted(k for k in keys if ref.metadata.get(k) != cand.metadata.get(k)),
            }
    keep = ("run_id", "experiment", "commit", "estimated_usd", "stopped")
    return {**{k: run.get(k) for k in keep}, "reference": reference, "runs": runs, "shifts": shifts}


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    summary = summarise(run_dir)
    (run_dir / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
