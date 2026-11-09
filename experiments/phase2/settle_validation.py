"""Calibration of the INCONCLUSIVE settle estimate, on fresh synthetic data.

For each undecided condition, `check` states an extension of the data and its chance to settle the
condition, averaged over the true values the data allows (hulder.gate.settle). This draws that
extension as fresh examples from the same generator, at the same threshold, and reads the condition
again. Across seeds, stated chances should match how often it settles.

The first estimator, which plugged in the observed value, missed its bar at commit ab8daaae61b7
(docs/results/phase2/settle.md). This is the second, on seeds that run never used. The bar is
committed before any run:

    source scripts/dev-env.sh
    uv run python experiments/phase2/settle_validation.py expected
    git add docs/results/phase2/settle_v2_expected.json && git commit
    OMP_NUM_THREADS=1 uv run python experiments/phase2/settle_validation.py run

`run` refuses unless the bar is committed and matches what the code recomputes. Each case and size
is written to .cache/settle-validation-v2/rows.jsonl as it finishes, and a rerun resumes from there.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experiments/phase4/probe"), str(ROOT / "experiments/phase1/probe")]

from resumable_rows import RowStore, run_rows
from synthetic import N_ATTACK, N_BENIGN, missed_attack_shift_for, unchanged_scorer

import hulder
from hulder.gate.check import AUROC_TOLERANCE, FNR_TOLERANCE, FPR_BAND_HIGH, _fnr, _fpr

EXPECTED = ROOT / "docs/results/phase2/settle_v2_expected.json"
RESULTS = ROOT / "docs/results/phase2/settle_v2.json"
WORK = ROOT / ".cache/settle-validation-v2"
CODE = (
    Path(__file__).resolve(),
    ROOT / "tests/synthetic.py",
    ROOT / "src/hulder/gate/settle.py",
    ROOT / "src/hulder/gate/check.py",
    ROOT / "src/hulder/gate/stats.py",
)

SEPARATION = 3.5  # synthetic.unchanged_scorer's default
SEEDS = range(2000, 3000)  # the first run used 0 to 999
EXTENSION_SEED_OFFSET = 3_000_000
# (benign, attack, target FPR). 180 per class runs at the quickstart's 5%, where PASS is reachable.
SIZES = {"calibration-sized": (N_BENIGN, N_ATTACK, 0.01), "small": (180, 180, 0.05)}
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MIN_SEEDS = 50
CALIBRATION_TOLERANCE = 0.10
WRONG_WAY_MAX = 0.01


def fnr_rise_shift(rise: float, target_fpr: float, separation: float = SEPARATION) -> float:
    """The attack-only logit shift that raises the population FNR by `rise` at the population threshold.
    Attacks are N(0, 1) and the threshold's logit is Φ⁻¹(1 − target) − separation."""
    t = norm.ppf(1 - target_fpr) - separation
    return float(t - norm.ppf(norm.cdf(t) + rise))


def cases(target: float) -> dict[str, dict]:
    """Each case's unchanged_scorer arguments at a target FPR, with the condition and side its
    population sits on: "pass" inside the tolerance, "fail" past it, None on the line."""
    return {
        "null": {"kwargs": {}, "side": {}},
        "FPR at 1.5x the target": {"kwargs": {"candidate_fpr": 1.5 * target}, "side": {"fpr": "pass"}},
        "FPR at 2x the target, on the line": {"kwargs": {"candidate_fpr": 2 * target}, "side": {"fpr": None}},
        "FPR at 2.5x the target": {"kwargs": {"candidate_fpr": 2.5 * target}, "side": {"fpr": "fail"}},
        "FNR rise of 0.04": {"kwargs": {"attack_shift": fnr_rise_shift(0.04, target)}, "side": {"fnr": "pass"}},
        "FNR rise of 0.05, on the line": {
            "kwargs": {"attack_shift": fnr_rise_shift(0.05, target)},
            "side": {"fnr": None},
        },
        "FNR rise of 0.06": {"kwargs": {"attack_shift": fnr_rise_shift(0.06, target)}, "side": {"fnr": "fail"}},
        "missed attacks to ΔAUROC -0.01": {
            "kwargs": {"missed_attack_shift": missed_attack_shift_for(-0.01, SEPARATION, target)},
            "side": {"delta_auroc": "pass"},
        },
        "missed attacks to ΔAUROC -0.02, on the line": {
            "kwargs": {"missed_attack_shift": missed_attack_shift_for(-0.02, SEPARATION, target)},
            "side": {"delta_auroc": None},
        },
        "missed attacks to ΔAUROC -0.025": {
            "kwargs": {"missed_attack_shift": missed_attack_shift_for(-0.025, SEPARATION, target)},
            "side": {"delta_auroc": "fail"},
        },
    }


def population_side(case: dict, condition: str) -> str | None:
    """A condition the case doesn't move sits inside its tolerance."""
    return case["side"].get(condition, "pass")


def expected_document() -> dict:
    return {
        "estimate": "for each undecided condition, the smallest extension up to ten times the data with at least"
        " an 80% chance to settle it, or ten times the data and its chance, averaged over the true values the"
        " data allows (hulder.gate.settle). ΔAUROC stops at 40,000 examples in all",
        "bars": {
            "calibration": f"per size, pooling every case and condition, the stated chances binned at {list(BINS)}:"
            f" in every bin with at least {MIN_SEEDS} seeds, the share settled after the extension is within"
            f" ±{CALIBRATION_TOLERANCE * 100:.0f} points of the bin's mean stated chance",
            "wrong way": f"per case, size and condition whose population sits on one side, with at least"
            f" {MIN_SEEDS} settled seeds: at most {WRONG_WAY_MAX:.0%} of them settle on the other side",
        },
        "not_claimed": "calibration for a truth exactly on a line. A chance averaged over plausible truths can"
        " hold on average and still miss for a truth on the line, where no amount of data settles it. The"
        " on-the-line cases are pooled into the bins like any other, and their own rates are reported",
        "reported_without_bar": [
            "each case's own calibration, by bin",
            "ΔAUROC extensions stopped by the 40,000-example cap",
            "the planted-fault rows, extended by drawing with replacement from the full split",
        ],
        "settled_means": "the condition's 95% bound has cleared its line on one side, computed as `check` does:"
        " the FPR and FNR bounds with the gate's own functions, ΔAUROC by a full check",
        "extension": f"fresh examples from the same generator and arguments, seed + {EXTENSION_SEED_OFFSET:,},"
        " at the original threshold. The FNR line is re-read on the extended data",
        "seeds": [SEEDS.start, SEEDS.stop],
        "bins": list(BINS),
        "min_seeds": MIN_SEEDS,
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "wrong_way_max": WRONG_WAY_MAX,
        "sizes": {name: {"n_benign": b, "n_attack": a, "target_fpr": t} for name, (b, a, t) in SIZES.items()},
        "cases": {name: {size: cases(t)[name] for size, (_, _, t) in SIZES.items()} for name in cases(0.01)},
    }


def side(d, condition: str, target: float) -> str | None:
    """How the data settles a condition, exactly as `check` computes it: "pass", "fail" or None.
    FPR and FNR read only their Clopper-Pearson bounds, so they skip the full check and its bootstrap."""
    benign = d.labels == 0
    if condition == "fpr":
        e, line = _fpr(d.candidate[benign], d.threshold), FPR_BAND_HIGH * target
    elif condition == "fnr":
        e = _fnr(d.candidate[~benign], d.threshold)
        line = _fnr(d.reference[~benign], d.threshold).value + FNR_TOLERANCE
    else:
        e = _check(d, target).delta_auroc
        line = -AUROC_TOLERANCE
        return "pass" if e.low >= line else "fail" if e.high < line else None
    return "pass" if e.high <= line else "fail" if e.low > line else None


def _check(d, target: float) -> hulder.Report:
    return hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=target)


def run_row(row: dict) -> dict:
    n_benign, n_attack, target = SIZES[row["size"]]
    kwargs = cases(target)[row["case"]]["kwargs"]
    seeds = []
    for seed in SEEDS:
        d = unchanged_scorer(seed, n_benign=n_benign, n_attack=n_attack, target_fpr=target, **kwargs)
        report = _check(d, target)
        entries = []
        for u in report.undecided:
            s = u.settle
            e = unchanged_scorer(
                seed + EXTENSION_SEED_OFFSET,
                n_benign=s.more_benign,
                n_attack=s.more_attack,
                target_fpr=target,
                threshold=d.threshold,
                **kwargs,
            )
            both = type(d)(
                np.r_[d.reference, e.reference], np.r_[d.candidate, e.candidate], np.r_[d.labels, e.labels], d.threshold
            )
            entries.append(
                {
                    "condition": u.condition,
                    "more_benign": s.more_benign,
                    "more_attack": s.more_attack,
                    "chance": s.chance,
                    "reached": s.reached,
                    "capped": s.capped,
                    "settled": side(both, u.condition, target),
                }
            )
        seeds.append(
            {
                "seed": seed,
                "verdict": report.verdict.value,
                "out_of_reach": [r.condition for r in report.out_of_reach],
                "undecided": entries,
            }
        )
    print(row["key"], sum(s["verdict"] == "INCONCLUSIVE" for s in seeds), "INCONCLUSIVE", flush=True)
    return {"key": row["key"], "seeds": seeds}


def bins_of(entries: list[dict]) -> list[dict]:
    out = []
    for lo, hi in pairwise(BINS):
        inside = [u for u in entries if lo <= u["chance"] < hi or (hi == 1.0 and u["chance"] == 1.0)]
        if not inside:
            continue
        stated = float(np.mean([u["chance"] for u in inside]))
        realised = float(np.mean([u["settled"] is not None for u in inside]))
        applies = len(inside) >= MIN_SEEDS
        out.append(
            {
                "bin": [lo, hi],
                "seeds": len(inside),
                "mean_stated": stated,
                "realised": realised,
                "bar_applies": applies,
                "meets_bar": abs(realised - stated) <= CALIBRATION_TOLERANCE if applies else None,
            }
        )
    return out


def wrong_way(row: dict, entries: list[dict]) -> list[dict]:
    target = SIZES[row["size"]][2]
    case = cases(target)[row["case"]]
    out = []
    for condition in ("fpr", "fnr", "delta_auroc"):
        expected = population_side(case, condition)
        settled = [u for u in entries if u["condition"] == condition and u["settled"] is not None]
        if expected is None or not settled:
            continue
        wrong = sum(u["settled"] != expected for u in settled)
        applies = len(settled) >= MIN_SEEDS
        out.append(
            {
                "case": row["case"],
                "size": row["size"],
                "condition": condition,
                "population_side": expected,
                "settled": len(settled),
                "wrong_way": wrong,
                "share": wrong / len(settled),
                "bar_applies": applies,
                "meets_bar": wrong / len(settled) <= WRONG_WAY_MAX if applies else None,
            }
        )
    return out


def summarise(rows: list[dict], raw: dict[str, dict]) -> dict:
    entries = {row["key"]: [u for s in raw[row["key"]]["seeds"] for u in s["undecided"]] for row in rows}
    calibration = {
        size: bins_of([u for row in rows if row["size"] == size for u in entries[row["key"]]]) for size in SIZES
    }
    per_case = [{"case": row["case"], "size": row["size"], "bins": bins_of(entries[row["key"]])} for row in rows]
    wrong = [w for row in rows for w in wrong_way(row, entries[row["key"]])]
    capped = [u for e in entries.values() for u in e if u["capped"]]
    barred = [b for bins in calibration.values() for b in bins if b["bar_applies"]] + [
        w for w in wrong if w["bar_applies"]
    ]
    return {
        "meets_bar": bool(barred) and all(b["meets_bar"] for b in barred),
        "calibration": calibration,
        "wrong_way": wrong,
        "per_case": per_case,
        "capped": {"seeds": len(capped), "settled": sum(u["settled"] is not None for u in capped)},
        "out_of_reach_seeds": {
            row["key"]: sum(bool(s["out_of_reach"]) for s in raw[row["key"]]["seeds"]) for row in rows
        },
    }


def _uncommitted(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path)], capture_output=True, text=True, check=True
    )
    return bool(status.stdout.strip())


def _identity() -> dict:
    digest = hashlib.sha256()
    for path in CODE:
        digest.update(path.read_bytes())
    return {"expected_sha256": hashlib.sha256(EXPECTED.read_bytes()).hexdigest(), "code_sha256": digest.hexdigest()}


def stage_expected() -> int:
    EXPECTED.write_text(json.dumps(expected_document(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {EXPECTED}; review it and commit it before the run")
    return 0


def stage_run(workers: int) -> int:
    if _uncommitted(EXPECTED):
        print(f"{EXPECTED.name} isn't committed. Commit the bar before running.", file=sys.stderr)
        return 1
    if json.loads(EXPECTED.read_text()) != json.loads(json.dumps(expected_document(), ensure_ascii=False)):
        print(f"{EXPECTED.name} differs from what the code recomputes.", file=sys.stderr)
        return 1
    from common import run_manifest

    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(EXPECTED)], capture_output=True, text=True, check=True
    ).stdout.strip()
    store = RowStore(
        WORK / "rows.jsonl", _identity(), extra={"expected_file_commit": commit, "run_manifest": run_manifest(seed=0)}
    )
    print(f"{len(store.done)} rows already written", flush=True)
    rows = [{"key": f"{case}/{size}", "case": case, "size": size} for case in cases(0.01) for size in SIZES]
    run_rows(rows, run_row, store, workers)
    report = {**store.extra, **summarise(rows, store.done)}
    RESULTS.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("meets_bar", "calibration", "capped")}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("expected", "run"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    return stage_expected() if args.stage == "expected" else stage_run(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
