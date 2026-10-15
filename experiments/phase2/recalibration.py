"""The recalibration study: binomial-tail and conformal thresholds on synthetic scores with exact
ground truth, and the promotion round trip (docs/design.md §6). Bounds were fixed before running
in tests/recalibration_study.py.

Run from the repo root:
    source scripts/dev-env.sh && uv run python experiments/phase2/recalibration.py

Writes docs/results/phase2/recalibration.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experiments/phase1/probe")]

from common import run_manifest
from recalibration_study import BOUNDS, CASES, PROMOTIONS, SIZES, TARGETS, bounds_hold, promotion_round_trip, study_cell

OUT = ROOT / "docs/results/phase2/recalibration.json"
SEEDS = range(10_000)
PROMOTION_SEEDS = range(1000)


def main() -> None:
    cells = {}
    for case in CASES:
        for n in SIZES:
            for target in TARGETS:
                cells[(case, n, target)] = cell = study_cell(case, n, target, SEEDS)
                bt = cell["binomial_tail"]
                print(
                    case,
                    n,
                    target,
                    {m: round(c.get("exceedance", float("nan")), 4) for m, c in cell.items()},
                    bt["refused"],
                )
    promotions = {}
    for p in PROMOTIONS:
        promotions[p] = promotion_round_trip(*p, PROMOTION_SEEDS)
        print("promotion", p, promotions[p])
    held = bounds_hold(cells, promotions)
    print("bounds hold:", held)
    report = {
        "seeds": [SEEDS.start, SEEDS.stop],
        "promotion_seeds": [PROMOTION_SEEDS.start, PROMOTION_SEEDS.stop],
        "bounds": BOUNDS,
        "bounds_hold": held,
        "cells": [{"case": c, "n_benign": n, "target_fpr": t, **r} for (c, n, t), r in cells.items()],
        "promotions": [
            {"n_benign": b, "n_attack": a, "target_fpr": t, "candidate_fpr": f, **r}
            for (b, a, t, f), r in promotions.items()
        ],
        "run_manifest": run_manifest(seed=SEEDS.start),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
