"""Coverage of the gate's ΔAUROC confidence interval on the missed-attack cases.

For each seed the true ΔAUROC is the population integral at that seed's fitted threshold
(tests/synthetic.py), so the threshold's scatter across seeds is not counted as a miss.
Coverage against the population value, which includes that scatter, is reported alongside.

Run from the repo root:
    source scripts/dev-env.sh && uv run python experiments/phase1/gate/coverage.py

Writes docs/results/phase1/delta_auroc_coverage.json.
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experiments/phase1/probe")]

from common import run_manifest
from synthetic import (
    N_ATTACK,
    N_BENIGN,
    missed_attack_delta_auroc,
    missed_attack_shift_for,
    unchanged_scorer,
)

import hulder

OUT = ROOT / "docs/results/phase1/delta_auroc_coverage.json"
SEEDS = range(1000)
DELTA_AUROCS = (-0.01, -0.02, -0.05, -0.1)


def _check_delta_auroc(reference, candidate, labels, seed, threshold) -> tuple[float, float]:
    """The ΔAUROC CI check reports, the bounds the verdict reads."""
    delta = hulder.check(reference, candidate, labels, threshold, 0.01, seed=seed).delta_auroc
    return delta.low, delta.high


METHODS: dict[str, Callable] = {"bootstrap": _check_delta_auroc}


def rate(k: int, n: int) -> dict:
    ci = binomtest(k, n).proportion_ci(method="exact")
    return {"rate": k / n, "ci95": [ci.low, ci.high]}


def main() -> None:
    results = {}
    for delta in DELTA_AUROCS:
        shift = missed_attack_shift_for(delta)
        truths, intervals, seconds = [], {m: [] for m in METHODS}, {m: 0.0 for m in METHODS}
        for seed in SEEDS:
            d = unchanged_scorer(seed, missed_attack_shift=shift)
            truths.append(missed_attack_delta_auroc(shift, threshold_logit=float(logit(d.threshold))))
            for name, method in METHODS.items():
                start = time.perf_counter()
                intervals[name].append(method(d.reference, d.candidate, d.labels, seed, d.threshold))
                seconds[name] += time.perf_counter() - start
        truth = np.array(truths)
        results[str(delta)] = {"per_seed_truth": {"mean": truth.mean(), "sd": truth.std(ddof=1)}}
        for name in METHODS:
            low, high = np.array(intervals[name]).T
            n = len(SEEDS)
            results[str(delta)][name] = {
                "coverage": rate(int(((low <= truth) & (truth <= high)).sum()), n),
                "low_above_truth": rate(int((low > truth).sum()), n),
                "high_below_truth": rate(int((high < truth).sum()), n),
                "coverage_of_population": rate(int(((low <= delta) & (delta <= high)).sum()), n),
                "mean_width": float((high - low).mean()),
                "seconds_per_call": seconds[name] / n,
            }
            print(delta, name, {k: v for k, v in results[str(delta)][name].items()}, flush=True)
    report = {
        "n_benign": N_BENIGN,
        "n_attack": N_ATTACK,
        "seeds": [SEEDS.start, SEEDS.stop],
        "results": results,
        "run_manifest": run_manifest(seed=SEEDS.start),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
