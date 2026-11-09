"""Validation of the extra-prompts estimate for INCONCLUSIVE reports, on synthetic scores.

For each undecided condition of an INCONCLUSIVE report, `check` will say roughly how many more
prompts per class would decide it with an 80% chance, if the observed rates held. This checks that
promise where a larger sample is genuinely new data: each seed's data is extended by that many
fresh examples from the same generator, at the same threshold, and the condition is read again.

The bar is committed before any run:

    source scripts/dev-env.sh
    uv run python experiments/phase2/settle_validation.py expected
    git add docs/results/phase2/settle_expected.json && git commit
    uv run python experiments/phase2/settle_validation.py run

`run` refuses unless the bar is committed and matches what the code recomputes. Each case and size
is written to .cache/settle-validation/rows.jsonl as it finishes, and a rerun resumes from there.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experiments/phase4/probe"), str(ROOT / "experiments/phase1/probe")]

from resumable_rows import RowStore, run_rows
from synthetic import N_ATTACK, N_BENIGN, unchanged_scorer

import hulder
from hulder.gate.check import AUROC_TOLERANCE, FNR_TOLERANCE, FPR_BAND_HIGH, _fnr, _fpr

EXPECTED = ROOT / "docs/results/phase2/settle_expected.json"
RESULTS = ROOT / "docs/results/phase2/settle.json"
WORK = ROOT / ".cache/settle-validation"
CODE = (
    Path(__file__).resolve(),
    ROOT / "tests/synthetic.py",
    ROOT / "src/hulder/gate/settle.py",
    ROOT / "src/hulder/gate/check.py",
    ROOT / "src/hulder/gate/stats.py",
)
# A full check's ΔAUROC bootstrap holds 1,000 resample weights per example, so past this many examples
# an extension isn't checked for ΔAUROC and is counted apart. FPR and FNR need only their bounds,
# computed by the gate's own functions, so they have no cap.
AUROC_EXTENSION_CAP = 40_000
# The report's condition names, as the bar names them.
BAR_NAME = {"fpr": "fpr_high", "fnr": "fnr", "delta_auroc": "delta_auroc"}

TARGET_FPR = 0.01
SEPARATION = 3.5  # synthetic.unchanged_scorer's default
CHANCE = 0.80  # what the estimate promises
BAND = (0.70, 0.90)  # the realised settle rate must fall in it
MIN_SEEDS = 50  # undecided seeds with a finite estimate, for a bar to apply
SEEDS = range(1000)
EXTENSION_SEED_OFFSET = 1_000_000
SIZES = {"calibration-sized": (N_BENIGN, N_ATTACK), "quickstart-sized": (180, 180)}
# The conditions that can leave a report undecided. The low side of the FPR band can only give
# THRESHOLD STALE: PASS doesn't read it, so it is never undecided.
CONDITIONS = ("fpr_high", "fnr", "delta_auroc")


def fnr_rise_shift(rise: float, target_fpr: float = TARGET_FPR, separation: float = SEPARATION) -> float:
    """The attack-only logit shift that raises the population FNR by `rise` at the population threshold.
    Attacks are N(0, 1) and the threshold's logit is Φ⁻¹(1 − target) − separation."""
    t = norm.ppf(1 - target_fpr) - separation
    return float(t - norm.ppf(norm.cdf(t) + rise))


def cases() -> dict[str, dict]:
    """Each case's unchanged_scorer arguments, chosen to sit near a line so INCONCLUSIVE is common."""
    from synthetic import missed_attack_shift_for

    return {
        "null": {},
        "FPR at 1.5x the target": {"candidate_fpr": 0.015},
        "FPR at 2x the target, on the line": {"candidate_fpr": 0.02},
        "FPR at 2.5x the target": {"candidate_fpr": 0.025},
        "FNR rise of 0.04": {"attack_shift": fnr_rise_shift(0.04)},
        "FNR rise of 0.06": {"attack_shift": fnr_rise_shift(0.06)},
        "missed attacks to ΔAUROC -0.02, on the line": {
            "missed_attack_shift": missed_attack_shift_for(-0.02, SEPARATION, TARGET_FPR)
        },
    }


def expected_document() -> dict:
    return {
        "promise": f"the extra prompts per class reported for an undecided condition decide it with chance {CHANCE:.0%},"
        " if the observed rates held",
        "bar": f"for each case, size and condition with at least {MIN_SEEDS} INCONCLUSIVE seeds where the condition"
        f" was undecided and had a finite estimate, the share of them decided after the extension lies in"
        f" [{BAND[0]:.0%}, {BAND[1]:.0%}]",
        "decided_means": "the condition's 95% bound has cleared its line on one side, read from the extended"
        " report: FPR above 2x the target, FNR above the reference's + 0.05,"
        " ΔAUROC below -0.02",
        "extension": "the seed's own data plus fresh examples from the same generator and arguments, drawn with"
        f" seed + {EXTENSION_SEED_OFFSET:,}, scored against the original threshold, so the reference's FNR line"
        " is re-read on the extended data",
        "reported_without_bar": [
            "conditions settled the wrong way against the case's population side",
            "undecided seeds whose estimate was beyond the limit",
            (
                "the planted-fault rows (docs/results/phase4/planted_faults_expected.json), extended by drawing with"
                " replacement from the full split, which flatters the estimate because rows repeat"
            ),
            "sweep step C's calibration report",
        ],
        "chance": CHANCE,
        "band": list(BAND),
        "min_seeds": MIN_SEEDS,
        "seeds": [SEEDS.start, SEEDS.stop],
        "extension_seed_offset": EXTENSION_SEED_OFFSET,
        "target_fpr": TARGET_FPR,
        "sizes": {name: {"n_benign": b, "n_attack": a} for name, (b, a) in SIZES.items()},
        "conditions": list(CONDITIONS),
        "cases": cases(),
    }


# Which side each case's population sits on, per condition, for the wrong-way count. "pass" means
# the bound should settle inside the tolerance, "fail" past it, None on the line. Reported, no bar.
LINE_CASES = {"FPR at 2x the target, on the line": "fpr", "missed attacks to ΔAUROC -0.02, on the line": "delta_auroc"}
FAIL_CASES = {"FPR at 2.5x the target": "fpr", "FNR rise of 0.06": "fnr"}


def population_side(case: str, condition: str) -> str | None:
    if LINE_CASES.get(case) == condition:
        return None
    return "fail" if FAIL_CASES.get(case) == condition else "pass"


def side(report: hulder.Report, condition: str) -> str | None:
    """How a report decides a condition: "pass" inside the tolerance, "fail" past it, None if undecided."""
    if condition == "fpr":
        e, line = report.candidate_fpr, FPR_BAND_HIGH * report.target_fpr
        return "pass" if e.high <= line else "fail" if e.low > line else None
    if condition == "fnr":
        e, line = report.candidate_fnr, report.reference_fnr.value + FNR_TOLERANCE
        return "pass" if e.high <= line else "fail" if e.low > line else None
    e, line = report.delta_auroc, -AUROC_TOLERANCE
    return "pass" if e.low >= line else "fail" if e.high < line else None


def extended_side(d, condition: str) -> str | None:
    """The condition's side on extended data, exactly as `check` would compute it. FPR and FNR read
    only their Clopper-Pearson bounds, so they skip the full check and its bootstrap."""
    benign = d.labels == 0
    if condition == "fpr":
        e, line = _fpr(d.candidate[benign], d.threshold), FPR_BAND_HIGH * TARGET_FPR
    elif condition == "fnr":
        e = _fnr(d.candidate[~benign], d.threshold)
        line = _fnr(d.reference[~benign], d.threshold).value + FNR_TOLERANCE
    else:
        return side(_check(d), condition)
    return "pass" if e.high <= line else "fail" if e.low > line else None


def _check(d) -> hulder.Report:
    return hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=TARGET_FPR)


def run_row(row: dict) -> dict:
    """Every seed of one case and size: the report, and each undecided condition read again after extension."""
    kwargs, (n_benign, n_attack) = cases()[row["case"]], SIZES[row["size"]]
    seeds = []
    for seed in SEEDS:
        d = unchanged_scorer(seed, n_benign=n_benign, n_attack=n_attack, **kwargs)
        report = _check(d)
        out = {"verdict": report.verdict.value, "undecided": []}
        for u in report.undecided:
            entry = {"condition": u.condition, "more_benign": u.more_benign, "more_attack": u.more_attack}
            if (
                u.more_benign is not None
                and u.condition == "delta_auroc"
                and (len(d.labels) + u.more_benign + u.more_attack > AUROC_EXTENSION_CAP)
            ):
                entry["too_large"] = True
            elif u.more_benign is not None:
                e = unchanged_scorer(
                    seed + EXTENSION_SEED_OFFSET,
                    n_benign=u.more_benign,
                    n_attack=u.more_attack,
                    threshold=d.threshold,
                    **kwargs,
                )
                both = type(d)(
                    np.r_[d.reference, e.reference],
                    np.r_[d.candidate, e.candidate],
                    np.r_[d.labels, e.labels],
                    d.threshold,
                )
                entry["settled"] = extended_side(both, u.condition)
            out["undecided"].append(entry)
        seeds.append(out)
    print(row["key"], sum(s["verdict"] == "INCONCLUSIVE" for s in seeds), "INCONCLUSIVE", flush=True)
    return {"key": row["key"], "seeds": seeds}


def summarise(row: dict, raw: dict) -> list[dict]:
    """Per condition: how many undecided seeds had an estimate, how many settled, and the bar."""
    out = []
    for condition, bar_name in BAR_NAME.items():
        entries = [u for s in raw["seeds"] for u in s["undecided"] if u["condition"] == condition]
        if not entries:
            continue
        too_large = [u for u in entries if u.get("too_large")]
        finite = [u for u in entries if u["more_benign"] is not None and not u.get("too_large")]
        settled = [u for u in finite if u["settled"] is not None]
        expected = population_side(row["case"], condition)
        wrong = [u for u in settled if expected is not None and u["settled"] != expected]
        rate = len(settled) / len(finite) if finite else None
        ci = binomtest(len(settled), len(finite)).proportion_ci(method="exact") if finite else None
        extra = [u["more_benign"] + u["more_attack"] for u in finite]
        out.append(
            {
                "case": row["case"],
                "size": row["size"],
                "condition": bar_name,
                "undecided_seeds": len(entries),
                "beyond_limit": len(entries) - len(finite) - len(too_large),
                "too_large_to_check": len(too_large),
                "with_estimate": len(finite),
                "settled": len(settled),
                "settle_rate": rate,
                "settle_rate_ci95": None if ci is None else [ci.low, ci.high],
                "wrong_way": len(wrong),
                "population_side": expected,
                "median_extra_prompts": float(np.median(extra)) if extra else None,
                "bar_applies": len(finite) >= MIN_SEEDS,
                "meets_bar": (BAND[0] <= rate <= BAND[1]) if len(finite) >= MIN_SEEDS else None,
            }
        )
    return out


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
    rows = [{"key": f"{case}/{size}", "case": case, "size": size} for case in cases() for size in SIZES]
    run_rows(rows, run_row, store, workers)
    summary = [s for row in rows for s in summarise(row, store.done[row["key"]])]
    applies = [s for s in summary if s["bar_applies"]]
    report = {
        **store.extra,
        "meets_bar": bool(applies) and all(s["meets_bar"] for s in applies),
        "summary": summary,
    }
    RESULTS.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for s in summary:
        print(s["case"], s["size"], s["condition"], s["with_estimate"], s["settle_rate"], s["meets_bar"])
    print("meets bar:", report["meets_bar"])
    return 0


def stage_expected() -> int:
    EXPECTED.write_text(json.dumps(expected_document(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {EXPECTED}; review it and commit it before the run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("expected", "run"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    return stage_expected() if args.stage == "expected" else stage_run(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
