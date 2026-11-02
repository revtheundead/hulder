"""D1 confirmation: the score stages of a finished run against the offline reference.

    source scripts/dev-env.sh && uv run python experiments/phase3/d1_scores_report.py .cache/modal-runs/<run id>

Reads each score stage's activations (d1_l4.score_stage), pairs them with the full capture's kept/ by
prompt id on the 2,101 calibration prompts, and scores them at the frozen primary probe's threshold.
Each stage is reported as decision flips by class, |Δscore|, |Δlogit| and rows bit-identical to kept/,
with the real hulder check beside it: kept/ as reference and experiment 2's shuffled rerun as
--noise-floor. Its verdict doesn't override the rule (score_outcome).

--smoke reads a run of d1_l4.py --smoke --confirmation instead, with the bundled demo reference,
rerun and probe, its floor measured from them.

Writes <run>/scores_report.json, refuses to run on a dirty tree, and runs once per run.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "phase4" / "probe"))
sys.path.insert(0, str(ROOT / "experiments" / "modal"))
import exp2_report as exp2
import full_capture_report as fcr
import smoke_lib as lib

import hulder

KEPT = Path(".cache/modal-runs/20261022T125644Z/kept")
SHUFFLED = Path(".cache/modal-runs/20261026T114520Z/shuffled")
EXP2 = ROOT / "docs" / "results" / "phase4" / "exp2.json"
SELECTION = ROOT / "docs" / "results" / "phase4" / "probe_qwen3_selection.json"
DEMO = ROOT / "src" / "hulder" / "data"
CELL = "18_-1"  # the primary, the frozen probe
STOCK, ZERO_DRAFT = "b_stock", "b_zero_draft"
REPORT_FILE = "scores_report.json"
MEASURES = ("flips", "mean_abs_dscore")


def floor_from(exp2_results: dict) -> dict:
    """The shuffled rerun against the control on the primary, as experiment 2 recorded it."""
    rerun = exp2_results["floor"]["primary"]["rerun"]
    return {"flips": rerun["flips"]["k"], "mean_abs_dscore": rerun["mean_abs_dscore"]}


def score_outcome(zero: list[dict], stock: list[dict], floor: dict) -> dict:
    """The rule fixed before the run, over zero-draft's stages against b_stock's. Pass: the median
    stage's mean |Δscore| is at most b_stock's highest, and every stage's flips at most b_stock's most.
    Fail: the lowest stage's mean |Δscore| is above b_stock's highest. Anything else needs a decision.
    So a pass says zero-draft moves scores no more than stock B does, not that either matches the
    reference. The floor doesn't decide, and each arm's stages above it are counted beside the outcome."""
    means = [run["mean_abs_dscore"] for run in zero]
    median, lowest = float(np.median(means)), min(means)
    highest = max(run["mean_abs_dscore"] for run in stock)
    most_flips = max(run["flips"] for run in zero)
    stock_most_flips = max(run["flips"] for run in stock)
    if lowest > highest:
        outcome = "fail"
    elif median <= highest and most_flips <= stock_most_flips:
        outcome = "pass"
    else:
        outcome = "needs decision"

    def above_floor(runs: list[dict]) -> int:
        return sum(any(run[m] > floor[m] for m in MEASURES) for run in runs)

    return {
        "outcome": outcome,
        "zero_draft_median_mean_abs_dscore": median,
        "zero_draft_lowest_mean_abs_dscore": lowest,
        "zero_draft_most_flips": most_flips,
        "b_stock_highest_mean_abs_dscore": highest,
        "b_stock_most_flips": stock_most_flips,
        "floor": floor,
        "zero_draft_stages_above_floor": above_floor(zero),
        "b_stock_stages_above_floor": above_floor(stock),
    }


def candidate_snapshot(reference: hulder.Snapshot, ids: list[str], rows: np.ndarray, metadata: dict) -> hulder.Snapshot:
    """A score stage's rows in the reference's order, with its labels and prompt id hash."""
    row = {str(i): k for k, i in enumerate(ids)}
    wanted = [str(i) for i in reference.ids]
    if missing := [i for i in wanted if i not in row]:
        raise ValueError(f"the stage lacks {len(missing)} of the reference's prompts, starting with {missing[0]}")
    drop = {"canary_sha256", "gpu_uuid", "gpu_driver"}
    kept = {k: v for k, v in reference.metadata.items() if k not in drop}
    return hulder.Snapshot(
        ids=np.asarray(reference.ids),
        labels=np.asarray(reference.labels),
        activations=np.asarray(rows)[[row[i] for i in wanted]],
        metadata={**kept, **metadata},
    )


def report(
    run_dir: Path,
    reference: hulder.Snapshot,
    rerun: hulder.Snapshot,
    probe_path: Path,
    floor: dict,
    *,
    recorded_floor: dict | None = None,
    run_check=exp2.run_check,
) -> dict:
    """Every stage of a finished run. Stops before scoring on a stopped run, a missing stage, or a
    rerun that no longer reproduces the floor the rule was fixed on."""
    results = json.loads((run_dir / "d1_l4.json").read_text())
    if results["stopped"]:
        return {"stopped_at": "run", "stops": [f"the run stopped: {results['stopped']}"]}
    stages = [run for run in results["runs"] if "score" in run]
    names = {f"r{run['round']}_{run['arm']}": run for run in stages}
    missing = [name for name in names if not (run_dir / name / "score.npz").exists()]
    if missing or not stages:
        return {"stopped_at": "run", "stops": [f"no score.npz for {missing or 'any stage'}"]}
    probe = hulder.Probe.load(probe_path)
    threshold = probe.metadata["threshold"]
    measured = exp2.paired_shift(probe, threshold, reference, rerun)
    measured_floor = {"flips": measured["flips"]["k"], "mean_abs_dscore": measured["mean_abs_dscore"]}
    if recorded_floor and measured_floor != recorded_floor:
        stop = f"the rerun doesn't reproduce the recorded floor: {measured_floor}, not {recorded_floor}"
        return {"stopped_at": "floor", "stops": [stop]}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    gpu = lib.gpu_metadata(manifest["measure"]["nvidia_smi"]) if manifest.get("measure", {}).get("nvidia_smi") else {}
    out: dict = {"threshold": threshold, "floor_measured": measured, "stages": {}}
    with tempfile.TemporaryDirectory() as tmp:
        paths = {"reference": Path(tmp) / "reference.npz", "rerun": Path(tmp) / "rerun.npz"}
        reference.save(paths["reference"])
        rerun.save(paths["rerun"])
        for name, run in names.items():
            with np.load(run_dir / name / "score.npz", allow_pickle=False) as saved:
                ids, rows = [str(i) for i in saved["ids"]], saved["activations"]
            metadata = {**gpu, "arm": run["arm"], "round": run["round"], "batch_size": run["score"]["concurrency"]}
            candidate = candidate_snapshot(reference, ids, rows, metadata)
            path = Path(tmp) / f"{name}.npz"
            candidate.save(path)
            out["stages"][name] = {
                "shift": exp2.paired_shift(probe, threshold, reference, candidate),
                "check": run_check(probe_path, paths["reference"], paths["rerun"], path),
            }
            path.unlink()

    def measures(arm: str) -> list[dict]:
        return [
            {
                "flips": out["stages"][name]["shift"]["flips"]["k"],
                "mean_abs_dscore": out["stages"][name]["shift"]["mean_abs_dscore"],
            }
            for name, run in names.items()
            if run["arm"] == arm
        ]

    out["outcome"] = score_outcome(measures(ZERO_DRAFT), measures(STOCK), floor)
    out["stopped_at"] = None
    return out


def refuse_to_rerun(run_dir: Path) -> None:
    if (run_dir / REPORT_FILE).exists():
        sys.exit(f"{run_dir / REPORT_FILE} exists; this report runs once per run")


def frozen_inputs() -> tuple[hulder.Snapshot, hulder.Snapshot, Path, dict]:
    """kept/ and the shuffled rerun on the calibration prompts, in kept/'s order, with the frozen
    primary probe, whose hash and threshold are checked, and the floor recorded by experiment 2."""
    selection = json.loads(SELECTION.read_text())
    probe_path, _, _ = exp2._frozen_probe(exp2.t.PROBES, CELL, selection)
    kept = hulder.Snapshot.load(KEPT / f"{CELL}.npz")
    splits = fcr.splits_by_id()
    ids = [str(i) for i in kept.ids if splits.get(str(i)) == "calibration"]
    tokens = dict(zip(*lib.load_token_ids(KEPT / lib.TOKEN_IDS_FILE), strict=True))
    reference = exp2.select(kept, ids, tokens)
    rerun = exp2.select(hulder.Snapshot.load(SHUFFLED / f"{CELL}.npz"), ids, tokens)
    return reference, rerun, probe_path, floor_from(json.loads(EXP2.read_text()))


def smoke_inputs() -> tuple[hulder.Snapshot, hulder.Snapshot, Path, dict]:
    reference = hulder.Snapshot.load(DEMO / "demo_fp16.npz")
    rerun = hulder.Snapshot.load(DEMO / "demo_fp16_run1.npz")
    probe_path = DEMO / "probe_demo.npz"
    probe = hulder.Probe.load(probe_path)
    shift = exp2.paired_shift(probe, probe.metadata["threshold"], reference, rerun)
    return reference, rerun, probe_path, {"flips": shift["flips"]["k"], "mean_abs_dscore": shift["mean_abs_dscore"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="a smoke run, against the bundled demo")
    args = parser.parse_args(argv)
    refuse_to_rerun(args.run_dir)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    )
    if status.stdout:
        sys.exit("the tracked tree has changes; commit them so the report's commit is what ran")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    reference, rerun, probe_path, floor = smoke_inputs() if args.smoke else frozen_inputs()
    result = {
        "commit": commit,
        **report(args.run_dir, reference, rerun, probe_path, floor, recorded_floor=None if args.smoke else floor),
    }
    (args.run_dir / REPORT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "floor_measured"}, indent=2, ensure_ascii=False)[:20000])
    return 0 if result["stopped_at"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
