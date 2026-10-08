"""Compare repeated fp16 captures of the same prompts, bitwise and through the main probe.

Each run is a fresh engine with an identical CaptureConfig (extract.py). Run from the repo
root with two or more capture directories:
    source scripts/dev-env.sh
    uv run python experiments/phase1/probe/compare_runs.py .cache/probe/fp16_run1 .cache/probe/fp16 ...

Writes docs/results/phase1/extraction_reruns.json.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from common import Activations, run_manifest

import hulder

PROBE = Path(".cache/probe/probe_main.npz")
OUT = Path("docs/results/phase1/extraction_reruns.json")
BATCH = 8  # CaptureConfig.batch_size; a batch is 8 consecutive prompts


def differing_prompts(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.zeros(a.shape[0], bool)
    for i in range(0, a.shape[0], chunk):
        x, y = np.asarray(a[i : i + chunk]), np.asarray(b[i : i + chunk])
        out[i : i + chunk] = (x.view(np.uint16) != y.view(np.uint16)).reshape(len(x), -1).any(axis=1)
    return out


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    runs = {d.name: Activations.load(d) for d in dirs}
    probe = hulder.Probe.load(PROBE)
    t = probe.metadata["threshold"]
    ids = list(next(iter(runs.values())).row_of)
    assert all(list(r.row_of) == ids for r in runs.values())
    scores = {name: hulder.apply_probe(probe, r.for_probe(ids, probe)) for name, r in runs.items()}

    pairs = {}
    for x, y in itertools.combinations(runs, 2):
        diff = differing_prompts(runs[x].array, runs[y].array)
        dscore = np.abs(scores[x] - scores[y])
        pairs[f"{x} vs {y}"] = {
            "prompts_differing": int(diff.sum()),
            "share_prompts_differing": float(diff.mean()),
            "batches_differing": len(np.unique(np.flatnonzero(diff) // BATCH)),
            "batches": -(-len(ids) // BATCH),
            "probe_abs_dscore": {
                "mean": float(dscore.mean()),
                "p99": float(np.quantile(dscore, 0.99)),
                "max": float(dscore.max()),
            },
            "decision_flips_at_threshold": int(((scores[x] > t) != (scores[y] > t)).sum()),
        }
        print(f"{x} vs {y}: {json.dumps(pairs[f'{x} vs {y}'])}")
    report = {
        "n_prompts": len(ids),
        "probe": {k: probe.metadata[k] for k in ("layer", "position", "threshold")},
        "runs": {
            name: {k: r.capture_manifest.get(k) for k in ("git_commit", "git_dirty", "elapsed_s")}
            for name, r in runs.items()
        },
        "pairs": pairs,
        "run_manifest": run_manifest(inputs=(PROBE,)),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
