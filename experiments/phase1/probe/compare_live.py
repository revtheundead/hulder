"""How far the live tutorial's captures are from the bundled demo snapshots, per run.

Run from the repo root, after `python -m hulder.capture.demo --out .cache/live_demo`:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/compare_live.py

Writes docs/results/phase1/live_demo/vs_bundled.json: the comparison, and a manifest with the
commit, packages and the live snapshots' hashes.
"""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
from common import run_manifest

import hulder
from hulder.capture.demo import BUNDLED

COMMAND = "source scripts/dev-env.sh && uv run python experiments/phase1/probe/compare_live.py"
LIVE = Path(".cache/live_demo")
OUT = Path("docs/results/phase1/live_demo/vs_bundled.json")


def main() -> None:
    data = files("hulder.data")
    probe = hulder.Probe.load(str(data / "probe_demo.npz"))
    threshold = probe.metadata["threshold"]
    result = {}
    for live_name, bundled_name in BUNDLED.items():
        live, bundled = hulder.Snapshot.load(LIVE / f"{live_name}.npz"), hulder.Snapshot.load(str(data / bundled_name))
        if list(live.ids) != list(bundled.ids):
            raise SystemExit(f"{live_name}: ids differ from {bundled_name}")
        a, b = live.activations.astype(np.float32), bundled.activations.astype(np.float32)
        rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
        s_live, s_bundled = hulder.apply_probe(probe, live.activations), hulder.apply_probe(probe, bundled.activations)
        benign = live.labels == 0
        result[live_name] = {
            "bundled": bundled_name,
            "bit_identical": bool(np.array_equal(live.activations, bundled.activations)),
            "relative_l2_median": float(np.median(rel)),
            "relative_l2_max": float(rel.max()),
            "mean_abs_score_difference": float(np.abs(s_live - s_bundled).mean()),
            "decisions_differing": int(((s_live > threshold) != (s_bundled > threshold)).sum()),
            "benign_flagged_live": int((s_live[benign] > threshold).sum()),
            "benign_flagged_bundled": int((s_bundled[benign] > threshold).sum()),
        }
    inputs = tuple(LIVE / f"{name}.npz" for name in BUNDLED)
    out = {"manifest": {"command": COMMAND, **run_manifest(inputs=inputs)}, "comparison": result}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
