"""The gate's own false-alarm and detection rates on synthetic data (README §7).

Uses the generators in tests/synthetic.py, sized like the S-Labs calibration set
(1,191 benign, 910 attack), with the threshold fitted on the reference sample.

Run from the repo root:
    source scripts/dev-env.sh && uv run python experiments/phase1/gate/calibration.py

Writes docs/results/phase1/gate_calibration.json.
"""

import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experiments/phase1/probe")]

from common import run_manifest
from synthetic import (
    FRESH_ENGINE_NOISE,
    FRESH_ENGINE_P_ZERO,
    N_ATTACK,
    N_BENIGN,
    blur_for,
    missed_attack_shift_for,
    reports,
    unchanged_scorer,
)

import hulder

OUT = ROOT / "docs/results/phase1/gate_calibration.json"
SEEDS = range(1000)
TARGET_FPR = 0.01
CANDIDATE_FPRS = (0.015, 0.02, 0.03, 0.05)
# Population ΔAUROC of the degraded candidates (gate_calibration.md has the derivation).
DELTA_AUROCS = (-0.01, -0.02, -0.05, -0.1)


def summarise(runs: list[hulder.Report]) -> dict:
    """Each verdict's rate with a 95% CI, and how many more benign INCONCLUSIVE runs asked for."""
    counts = Counter(r.verdict for r in runs)
    out = {}
    for verdict in hulder.Verdict:
        ci = binomtest(counts[verdict], len(runs)).proportion_ci(method="exact")
        out[verdict.value] = {"rate": counts[verdict] / len(runs), "ci95": [ci.low, ci.high]}
    more = [r.more_benign_needed for r in runs if r.verdict is hulder.Verdict.INCONCLUSIVE]
    known = [m for m in more if m is not None]
    out["inconclusive_more_benign_needed"] = {
        "median": float(np.median(known)) if known else None,
        "quartiles": [float(q) for q in np.quantile(known, [0.25, 0.75])] if known else None,
        "n_none": len(more) - len(known),
    }
    return out


def main() -> None:
    cases = {
        "null": unchanged_scorer,
        "fresh_engine_noise": partial(unchanged_scorer, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO),
        **{f"candidate_fpr_{x}": partial(unchanged_scorer, candidate_fpr=x) for x in CANDIDATE_FPRS},
        **{
            f"missed_attack_delta_auroc_{d}": partial(unchanged_scorer, missed_attack_shift=missed_attack_shift_for(d))
            for d in DELTA_AUROCS
        },
        **{f"blur_delta_auroc_{d}": partial(unchanged_scorer, noise=blur_for(d)) for d in DELTA_AUROCS},
    }
    results = {}
    for name, make in cases.items():
        results[name] = summarise(reports(make, SEEDS, TARGET_FPR))
        print(name, {v: round(results[name][v.value]["rate"], 3) for v in hulder.Verdict})
    report = {
        "n_benign": N_BENIGN,
        "n_attack": N_ATTACK,
        "target_fpr": TARGET_FPR,
        "seeds": [SEEDS.start, SEEDS.stop],
        "degradation_sizes": {
            str(d): {"missed_attack_shift": missed_attack_shift_for(d), "blur_noise_sd": blur_for(d)}
            for d in DELTA_AUROCS
        },
        "results": results,
        "run_manifest": run_manifest(seed=SEEDS.start),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
