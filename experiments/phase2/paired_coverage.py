"""Coverage of the candidate paired intervals, on synthetic paired 2×2 tables with known truth.

The paired FPR and FNR conditions (docs/design.md §6) bound the change on the same examples,
(b − c) / n, from b newly flagged and c newly unflagged. This draws tables from a multinomial with
known cell probabilities and counts how often each interval in paired_intervals.py misses the true
difference, at each end separately, since the gate reads one end per condition. The bar and the
choice rule are committed before any run:

    source scripts/dev-env.sh
    uv run python experiments/phase2/paired_coverage.py expected --run first
    git add docs/results/phase2/paired_coverage_expected.json && git commit
    uv run python experiments/phase2/paired_coverage.py run --run first

There are two runs, each with its own intervals, seeds, bar file and results. The first tested three
intervals and none met the bar (docs/results/phase2/paired_coverage.md). The second tests one more,
Newcombe's interval with continuity correction, chosen after seeing the first run's cells, against the
same bar on seeds the first didn't use.

`run` refuses unless the bar is committed and matches what the code recomputes. Each cell is written
to the run's rows.jsonl under .cache/ as it finishes, and a rerun resumes from there.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "experiments/phase4/probe")]

from paired_intervals import ALPHA, CANDIDATES
from resumable_rows import RowStore, run_rows

RESULTS_DIR = ROOT / "docs/results/phase2"
CODE = (Path(__file__).resolve(), Path(__file__).resolve().parent / "paired_intervals.py")

# Fresh seeds for each run. Settle used 0 to 999 and 2,000 to 2,999, the flip check and planted faults
# 0 to 999, and the shift note keeps 1,000 to 1,999. The paired rule's other runs hold 3,000 to 3,999,
# 5,000 to 6,999 and 8,000 to 8,999 (paired_rule_expected.json), so the second run takes 9,000 to 9,999.
RUNS = {
    "first": {
        "intervals": ("exact_conditional", "tango", "newcombe"),
        "seeds": range(7000, 8000),
        "name": "paired_coverage",
    },
    "continuity_corrected": {
        "intervals": ("newcombe_cc",),
        "seeds": range(9000, 10000),
        "name": "paired_coverage_cc",
    },
}
DRAWS_PER_SEED = 10  # 10,000 tables per cell
# The gate's sizes: the quickstart's 180 per class, the Qwen3-4B calibration split's 1,191 benign and
# 910 attacks, and its test split's 1,050 benign.
SIZES = (180, 910, 1050, 1191)
# Share flagged by both runs: about the FPR at a 1% or 5% target, and an FNR near the primary probe's.
BOTH = (0.01, 0.05, 0.15)
# Share flagged by one run only, each way. 0.5% to 1% at n = 180 is where 0 discordant is common.
ONE_WAY = (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05)
CELL_MAX = 0.035
MEAN_MAX = 0.0275
LOW_DISCORDANCE = {"n": 180, "total_discordant": (0.005, 0.01)}
TOLERANCE = 1e-12


def expected_path(run: str) -> Path:
    return RESULTS_DIR / f"{RUNS[run]['name']}_expected.json"


def results_path(run: str) -> Path:
    return RESULTS_DIR / f"{RUNS[run]['name']}.json"


def work_dir(run: str) -> Path:
    return ROOT / ".cache" / RUNS[run]["name"].replace("_", "-")


def cells(run: str) -> list[dict]:
    return [
        {"key": f"n={n}/a={a}/b={b}/c={c}", "run": run, "n": n, "a": a, "b": b, "c": c}
        for n, a, b, c in product(SIZES, BOTH, ONE_WAY, ONE_WAY)
    ]


def expected_document(run: str) -> dict:
    document = _first_document(run)
    if run == "continuity_corrected":
        document["chosen_after_the_first_run"] = (
            "this candidate was chosen after seeing the first run's cells (paired_coverage.md), where"
            " Newcombe's interval without continuity correction missed the bar in 5 cell-ends at n of 1,050"
            " and 1,191. It is the only candidate in this run, the bar is unchanged and the seeds are new"
        )
        document["choice"] = (
            "if it meets both bars, the paired FPR and FNR conditions use it and the gate checks run on their"
            " committed seeds. If not, the paired rule doesn't ship, the miss is published and item 2 closes"
        )
        document["reported_without_bar"][1] = (
            "the cells at n = 180 whose total discordant rate is 0.5% to 1%, listed as in the first run"
        )
    return document


def _first_document(run: str) -> dict:
    spec = RUNS[run]
    return {
        "question": "how often each candidate 95% interval for a paired difference of proportions misses the"
        " true difference, at each end, on the gate's sizes",
        "intervals": list(spec["intervals"]),
        "alpha_two_sided": ALPHA,
        "truth": "b − c, the cell probabilities of flagged by the candidate only and by the reference only",
        "miss": "low end: the lower bound is above the truth. High end: the upper bound is below it."
        " The gate reads the lower bound for THRESHOLD STALE and the upper bound for PASS",
        "draws": "10,000 tables per cell from a multinomial over (both, candidate only, reference only, neither),"
        " 10 for each seed with numpy default_rng([seed, n, a·10⁴, b·10⁴, c·10⁴])",
        "seeds": [spec["seeds"].start, spec["seeds"].stop],
        "grid": {
            "n": list(SIZES),
            "both": list(BOTH),
            "candidate_only": list(ONE_WAY),
            "reference_only": list(ONE_WAY),
        },
        "bars": {
            "cell": f"in every cell, each end misses in at most {CELL_MAX:.1%} of tables, against a nominal 2.5%",
            "mean": f"over all cells, each end's mean miss rate is at most {MEAN_MAX:.2%}",
        },
        "choice": "of the intervals meeting both bars, the one with the smallest mean width over all cells is used"
        " by the paired FPR and FNR conditions. If none meets both, the paired rule doesn't ship and the miss is"
        " published",
        "reported_without_bar": [
            "every cell's miss rates, widths and share of tables with no discordant example",
            (
                "the cells at n = 180 whose total discordant rate is 0.5% to 1%, listed per interval, where 0"
                " discordant examples are common and the exact conditional interval has no width"
            ),
        ],
        "cell_max": CELL_MAX,
        "mean_max": MEAN_MAX,
        "low_discordance": LOW_DISCORDANCE,
    }


def tables(cell: dict, seeds: range) -> np.ndarray:
    n = cell["n"]
    probs = [cell["a"], cell["b"], cell["c"], 1 - cell["a"] - cell["b"] - cell["c"]]
    tag = [round(cell[k] * 10_000) for k in ("a", "b", "c")]
    return np.concatenate(
        [np.random.default_rng([seed, n, *tag]).multinomial(n, probs, size=DRAWS_PER_SEED) for seed in seeds]
    )


def run_cell(cell: dict) -> dict:
    spec = RUNS[cell["run"]]
    drawn = tables(cell, spec["seeds"])
    truth = cell["b"] - cell["c"]
    counts = Counter(map(tuple, drawn[:, :3].tolist()))
    total = sum(counts.values())
    out = {
        "key": cell["key"],
        "tables": total,
        "no_discordant": sum(v for (a, b, c), v in counts.items() if b + c == 0) / total,
    }
    for name in spec["intervals"]:
        interval = CANDIDATES[name]
        low_miss = high_miss = width = 0.0
        for (a, b, c), weight in counts.items():
            low, high = interval(a, b, c, cell["n"])
            low_miss += weight * (low > truth + TOLERANCE)
            high_miss += weight * (high < truth - TOLERANCE)
            width += weight * (high - low)
        out[name] = {"low_miss": low_miss / total, "high_miss": high_miss / total, "mean_width": width / total}
    return out


def summarise(rows: list[dict], done: dict, intervals: tuple[str, ...]) -> dict:
    results = [{**row, **done[row["key"]]} for row in rows]
    per_interval = {}
    for name in intervals:
        worst = {end: max(results, key=lambda r: r[name][end]) for end in ("low_miss", "high_miss")}
        means = {end: float(np.mean([r[name][end] for r in results])) for end in ("low_miss", "high_miss")}
        per_interval[name] = {
            "worst_cell": {end: {"key": w["key"], "rate": w[name][end]} for end, w in worst.items()},
            "cells_over_cell_max": {
                end: sum(r[name][end] > CELL_MAX for r in results) for end in ("low_miss", "high_miss")
            },
            "mean_miss": means,
            "mean_width": float(np.mean([r[name]["mean_width"] for r in results])),
            "meets_bar": all(r[name][end] <= CELL_MAX for r in results for end in ("low_miss", "high_miss"))
            and all(m <= MEAN_MAX for m in means.values()),
        }
    meeting = [name for name in intervals if per_interval[name]["meets_bar"]]
    chosen = min(meeting, key=lambda name: per_interval[name]["mean_width"]) if meeting else None
    lo, hi = LOW_DISCORDANCE["total_discordant"]
    low_discordance = [
        {k: r[k] for k in ("key", "no_discordant", *intervals)}
        for r in results
        if r["n"] == LOW_DISCORDANCE["n"] and lo - 1e-9 <= r["b"] + r["c"] <= hi + 1e-9
    ]
    return {"per_interval": per_interval, "chosen": chosen, "low_discordance_n180": low_discordance, "cells": results}


def _uncommitted(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path)], capture_output=True, text=True, check=True
    )
    return bool(status.stdout.strip())


def _identity(run: str) -> dict:
    digest = hashlib.sha256()
    for path in CODE:
        digest.update(path.read_bytes())
    expected = hashlib.sha256(expected_path(run).read_bytes()).hexdigest()
    return {"run": run, "expected_sha256": expected, "code_sha256": digest.hexdigest()}


def stage_expected(run: str) -> int:
    path = expected_path(run)
    path.write_text(json.dumps(expected_document(run), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}; review it and commit it before the run")
    return 0


def stage_run(run: str, workers: int) -> int:
    expected = expected_path(run)
    if not expected.exists() or _uncommitted(expected):
        print(f"{expected.name} isn't committed. Commit the bar before running.", file=sys.stderr)
        return 1
    if json.loads(expected.read_text()) != json.loads(json.dumps(expected_document(run), ensure_ascii=False)):
        print(f"{expected.name} differs from what the code recomputes.", file=sys.stderr)
        return 1
    if _uncommitted(Path(__file__).resolve()) or _uncommitted(CODE[1]):
        print("the coverage code isn't committed. Commit it before running.", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT / "experiments/phase1/probe"))
    from common import run_manifest

    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(expected)], capture_output=True, text=True, check=True
    ).stdout.strip()
    spec = RUNS[run]
    store = RowStore(
        work_dir(run) / "rows.jsonl",
        _identity(run),
        extra={"expected_file_commit": commit, "run_manifest": run_manifest(seed=spec["seeds"].start)},
    )
    print(f"{len(store.done)} cells already written", flush=True)
    rows = cells(run)
    run_rows(rows, run_cell, store, workers)
    report = {**store.extra, **summarise(rows, store.done, spec["intervals"])}
    results_path(run).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"chosen": report["chosen"], **{k: v["meets_bar"] for k, v in report["per_interval"].items()}}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("expected", "run"))
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    return stage_expected(args.run) if args.stage == "expected" else stage_run(args.run, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
