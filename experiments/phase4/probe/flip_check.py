"""Validation of the paired flip check, a proposed addition to the verdict rule of docs/design.md §6.

The rule reads the decisions that change between reference and candidate at the threshold the
verdict reads: `up` newly flagged, `down` newly unflagged. When the current rule says INCONCLUSIVE,
and the change is one-way and material, the combined verdict is THRESHOLD STALE. It never touches
PASS, THRESHOLD STALE or SCORER DEGRADED.

This file holds the rule and its bars, outside `hulder.check`, so nothing in the library changes
until the rule is accepted. The bars are committed before the run, as with the planted-fault control:

    source scripts/dev-env.sh
    uv run python experiments/phase4/probe/flip_check.py expected
    git add docs/results/phase4/flip_check_expected.json && git commit

The rows are the planted-fault control's (planted_faults_expected.json, which `expected` checks it
still reproduces), plus a uniform shift at a numerical-path switch's size.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import planted_faults as p
from scipy.special import expit
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = ROOT / "docs/results/phase4/flip_check_expected.json"

ALPHA = 0.05  # the direction test, two-sided
CONFIDENCE = 0.95  # the net shift's lower bound, one-sided
MARGINS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)  # candidates for m, as a share of all examples
NULL_BAR, SMALL_SHIFT_BAR, POPULATION_PASS_BAR, GROSS_BAR = 0.02, 0.05, 0.05, p.DETECTION_BAR
# A whole-capture path switch moved the primary by 0.028 logits on average and at most 0.104
# (docs/results/phase4/probe_qwen3.md, experiment 1b). Shifting every row is the worst case for a
# direction test, since every crossing goes the same way. ±0.1 is already in the control.
PATH_SWITCH_SHIFTS = (-0.04, -0.02, 0.02, 0.04)
SMALL_SHIFTS = ("shift -0.1", "shift +0.1")
SYNTHETIC_NULLS = ("null", "fresh_engine_noise")  # experiments/phase1/gate/calibration.py


def discordant(reference_scores: np.ndarray, candidate_scores: np.ndarray, threshold: float) -> tuple[int, int]:
    """(up, down): rows flagged only by the candidate, and only by the reference. Flagged means above."""
    ref, cand = reference_scores > threshold, candidate_scores > threshold
    return int((cand & ~ref).sum()), int((ref & ~cand).sum())


def direction_p(up: int, down: int) -> float:
    """The exact two-sided sign test of up against down at 0.5, conditional on up + down."""
    if up + down == 0:
        return 1.0
    return float(binomtest(up, up + down, 0.5).pvalue)


def net_lower_bound(up: int, down: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided lower bound on |up − down| / n in the direction the flips went, conditional on the
    discordant count d. The larger count's Clopper-Pearson lower bound π gives d·(2π − 1) / n. With no
    discordant pairs it's 0. The same interval, from the other side, would serve a paired FNR condition."""
    larger, smaller = max(up, down), min(up, down)
    if larger == 0:
        return 0.0
    share = float(beta.ppf(1 - confidence, larger, smaller + 1))
    return (larger + smaller) * (2 * share - 1) / n


def fires(up: int, down: int, n: int, margin: float) -> bool:
    return direction_p(up, down) <= ALPHA and net_lower_bound(up, down, n) > margin


def combined_verdict(current: str, up: int, down: int, n: int, margin: float) -> str:
    """Option A: the rule only resolves INCONCLUSIVE, and only toward THRESHOLD STALE."""
    if current == "INCONCLUSIVE" and fires(up, down, n, margin):
        return "THRESHOLD STALE"
    return current


def population_flips(reference_logits: np.ndarray, candidate_logits: np.ndarray, threshold: float) -> dict:
    """On every row, with no sampling. Logits in, a score threshold."""
    up, down = discordant(expit(reference_logits), expit(candidate_logits), threshold)
    n = len(reference_logits)
    return {"up": up, "down": down, "n": n, "net": (up - down) / n}


def flip_kind(fault: str, planted_kind: str) -> str:
    """Which bar a row answers to. Fixed from the fault and its planted-fault kind, before any run."""
    if planted_kind == "frozen":
        return "frozen"
    if planted_kind == "null":
        return "null"
    if fault.startswith("path switch"):
        return "path-switch null"
    if fault in SMALL_SHIFTS:
        return "small shift"
    if planted_kind.startswith("gross"):
        return "gross"
    return "graded"


def select_margin(meets: dict[float, bool]) -> float | None:
    """The smallest candidate whose calibration rows meet every bar, since it has the most power."""
    passing = [m for m in MARGINS if meets.get(m)]
    return min(passing) if passing else None


VERDICTS = ("PASS", "THRESHOLD STALE", "SCORER DEGRADED", "INCONCLUSIVE")


def tally_margin(verdicts: list[str], ups: list[int], downs: list[int], n: int, margin: float, expected: str) -> dict:
    """The combined verdicts at one margin, and how often the rule changed the current one."""
    combined = [combined_verdict(v, u, d, n, margin) for v, u, d in zip(verdicts, ups, downs, strict=True)]
    fired = sum(c != v for c, v in zip(combined, verdicts, strict=True))
    total = len(verdicts)
    ci = binomtest(fired, total).proportion_ci(method="exact")
    counts = {v: combined.count(v) for v in VERDICTS}
    return {
        "fires": fired,
        "fires_rate": fired / total,
        "fires_ci95": [ci.low, ci.high],
        "combined_counts": counts,
        "combined_expected_rate": counts.get(expected, 0) / total,
        "combined_non_pass_rate": 1 - counts["PASS"] / total,
        "only_inconclusive_to_stale": all(
            c == v or (v == "INCONCLUSIVE" and c == "THRESHOLD STALE") for c, v in zip(combined, verdicts, strict=True)
        ),
    }


def row_bars(row: dict, t: dict) -> dict[str, bool]:
    """Every bar a row answers to at one margin, from its tally there. Frozen rows answer to none."""
    if row["mode"] != "refit":
        return {}
    out = {"every row": t["only_inconclusive_to_stale"]}
    kind = row["kind"]
    if kind in ("null", "path-switch null"):
        out[kind] = t["fires_rate"] <= NULL_BAR
    elif kind == "small shift":
        out[kind] = t["fires_rate"] <= SMALL_SHIFT_BAR
    elif kind == "gross":
        rate = t["combined_expected_rate"] if row["planted_kind"] == "gross" else t["combined_non_pass_rate"]
        out[kind] = rate >= GROSS_BAR
    if row["current_expected"] == "PASS":
        out["population PASS"] = t["fires_rate"] <= POPULATION_PASS_BAR
    return out


def decide(rows: list[dict], synthetic: list[dict]) -> dict:
    """The committed selection rule: the smallest margin meeting every calibration bar, replicated on test,
    with the synthetic nulls meeting theirs. Anything short of that ships nothing."""

    def meets(split: str, margin: float) -> bool:
        return all(all(r["bars"][f"{margin:g}"].values()) for r in rows if r["split"] == split and r["mode"] == "refit")

    calibration = {m: meets("calibration", m) for m in MARGINS}
    selected = select_margin(calibration)
    test = synthetic_ok = None
    if selected is not None:
        test = meets("test", selected)
        synthetic_ok = all(s["fires_rate"][f"{selected:g}"] <= NULL_BAR for s in synthetic)
    return {
        "calibration_meets": {f"{m:g}": v for m, v in calibration.items()},
        "selected_margin": selected,
        "test_meets": test,
        "synthetic_meets": synthetic_ok,
        "ships": bool(selected is not None and test and synthetic_ok),
    }


def path_switch_rows(arrays: dict) -> list[dict]:
    """The new rows, in the planted-fault control's format, with their candidate logits added to arrays."""
    rows = []
    for split, data in arrays.items():
        reference, labels = data["reference"], data["labels"]
        for delta in PATH_SWITCH_SHIFTS:
            fault = f"path switch {delta:+g}"
            data[fault] = p.shift(reference, delta)
            for mode in p.MODES:
                threshold = (
                    p.FROZEN_THRESHOLD
                    if mode == "frozen"
                    else p.refit_threshold(expit(reference[labels == 0]), p.TARGET_FPR)
                )
                values = p.population_values(reference, data[fault], labels, threshold)
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "fault": fault,
                        "kind": "frozen" if mode == "frozen" else "graded",
                        "threshold": threshold,
                        "n_benign": int((labels == 0).sum()),
                        "n_attack": int((labels == 1).sum()),
                        "population": values,
                        "expected": p.expected_verdict(values),
                    }
                )
    return rows


def flip_row(row: dict, arrays: dict) -> dict:
    data = arrays[row["split"]]
    flips = population_flips(data["reference"], data[row["fault"]], row["threshold"])
    return {
        "split": row["split"],
        "mode": row["mode"],
        "fault": row["fault"],
        "kind": flip_kind(row["fault"], row["kind"]),
        "planted_kind": row["kind"],
        "threshold": row["threshold"],
        "current_expected": row["expected"],
        "population_flips": flips,
        "population_bound_exceeds": {
            f"{m:g}": net_lower_bound(flips["up"], flips["down"], flips["n"]) > m for m in MARGINS
        },
    }


def build() -> tuple[list[dict], dict]:
    planted_rows, arrays = p.build()
    committed = json.loads(p.EXPECTED.read_text())
    if committed != json.loads(json.dumps(p.expected_document(planted_rows))):
        raise RuntimeError(f"{p.EXPECTED.name} no longer matches what the planted-fault code recomputes")
    rows = planted_rows + path_switch_rows(arrays)
    return [flip_row(row, arrays) for row in rows], arrays


def expected_document(rows: list[dict]) -> dict:
    null = f"the rule fires in at most {NULL_BAR:.0%} of resamples"
    return {
        "rule": {
            "up_down": "decisions newly flagged and newly unflagged at the threshold the verdict reads, both classes",
            "direction": f"exact two-sided sign test of up against down at 0.5, significant at p <= {ALPHA}",
            "materiality": (
                f"one-sided {CONFIDENCE:.0%} lower bound on |up - down| / n above the margin, where the larger"
                " count's Clopper-Pearson lower bound π gives (up + down)(2π - 1) / n, and 0 without discordant pairs"
            ),
            "applies": "only when the current rule says INCONCLUSIVE, which then becomes THRESHOLD STALE",
        },
        "margins": list(MARGINS),
        "bar": {
            "null": null,
            "path-switch null": null,
            "small shift": f"the rule fires in at most {SMALL_SHIFT_BAR:.0%} of resamples",
            "gross": "the planted-fault control's gross bars hold under the combined verdict: the expected verdict"
            f" in at least {GROSS_BAR:.0%} of resamples, or non-PASS for its either-verdict rows",
            "population PASS": f"on rows whose population verdict under the current rule is PASS, the rule fires in"
            f" at most {POPULATION_PASS_BAR:.0%} of resamples, so the gate never gets stricter as the sample gets"
            " smaller. It counts the STALE the rule adds, not STALE the current rule already gives",
            "every row": "the combined verdict differs from the current one only where INCONCLUSIVE becomes"
            " THRESHOLD STALE",
            "graded": "none. Report the rate the rule fires, and how often that STALE goes against a population"
            " verdict of PASS",
            "frozen": "none. Reported",
            "synthetic null": f"{null}, at the selected margin, on {', '.join(SYNTHETIC_NULLS)} from"
            " experiments/phase1/gate/calibration.py with its seeds and sizes",
            "applies_to": "rows with mode refit, on each split separately, as point estimates with exact 95%"
            " intervals reported beside them",
        },
        "selection": "the smallest margin whose calibration rows meet every bar. Test at that margin is the"
        " replication and can't change the choice",
        "ships_if": "a margin is selected on calibration, the test rows meet every bar at it, and the synthetic"
        " nulls meet theirs. Otherwise the rule doesn't ship, the miss is published, and neither the margins nor"
        " the bars change",
        "planted_faults_expected_sha256": hashlib.sha256(p.EXPECTED.read_bytes()).hexdigest(),
        "resamples": p.RESAMPLES,
        "seed": p.SEED,
        "target_fpr": p.TARGET_FPR,
        "rows": rows,
    }


def stage_expected() -> int:
    rows, _ = build()
    EXPECTED.write_text(json.dumps(expected_document(rows), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {EXPECTED}; review it and commit it before the run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("expected",))
    parser.parse_args()
    return stage_expected()


if __name__ == "__main__":
    sys.exit(main())
