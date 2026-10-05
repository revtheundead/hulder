"""The gate's own false-alarm and detection rates on synthetic data (README §7).

Uses the generators in tests/synthetic.py, sized like the S-Labs calibration set
(1,191 benign, 910 attack), with the threshold fitted on the reference sample.

Run from the repo root:
    source scripts/dev-env.sh && uv run python experiments/phase1/gate/calibration.py

Writes docs/results/phase1/gate_calibration.json.
"""

import json
import sys
from functools import partial
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experiments/phase1/probe")]

from common import run_manifest  # noqa: E402
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, N_ATTACK, N_BENIGN, unchanged_scorer, verdict_rates  # noqa: E402

OUT = ROOT / "docs/results/phase1/gate_calibration.json"
SEEDS = range(1000)
TARGET_FPR = 0.01
CANDIDATE_FPRS = (0.015, 0.02, 0.03, 0.05)


def with_ci(rates: dict) -> dict:
    out = {}
    for verdict, rate in rates.items():
        k = round(rate * len(SEEDS))
        ci = binomtest(k, len(SEEDS)).proportion_ci(method="exact")
        out[verdict] = {"rate": rate, "ci95": [ci.low, ci.high]}
    return out


def main() -> None:
    cases = {
        "null": unchanged_scorer,
        "fresh_engine_noise": partial(unchanged_scorer, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO),
        **{f"candidate_fpr_{x}": partial(unchanged_scorer, candidate_fpr=x) for x in CANDIDATE_FPRS},
    }
    results = {}
    for name, make in cases.items():
        results[name] = with_ci(verdict_rates(make, SEEDS, TARGET_FPR))
        print(name, {v: round(r["rate"], 3) for v, r in results[name].items()})
    report = {
        "n_benign": N_BENIGN,
        "n_attack": N_ATTACK,
        "target_fpr": TARGET_FPR,
        "seeds": [SEEDS.start, SEEDS.stop],
        "results": results,
        "run_manifest": run_manifest(seed=SEEDS.start),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
