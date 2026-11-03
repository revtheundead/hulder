"""The Phase 3 overhead table's score stages against the offline reference.

    source scripts/dev-env.sh && uv run python experiments/phase3/overhead_report.py .cache/modal-runs/<run id> --model qwen3
    uv run python experiments/phase3/overhead_report.py .cache/modal-runs/<run id> --model 1.5b

Reads each stage's scores (d1_l4.reply_score_stage) and runs hulder.check_snapshots on them, with the
reference scored by the same probe and its rerun as the noise floor. The Qwen3-4B reference is the full
capture's kept/ with the frozen primary probe, and the small model's is the bundled demo with its probe,
captured on an RTX 2070. So any difference from the 1.5B reference includes the GPU.

Prefix caching is declared a change under test, so a verdict on a caching-on stage reads as one on that
serving change. Unique prompts share little beyond the chat template, so caching-on stages test whether
caching moves scores, not how it behaves on a shared-prefix workload.

Writes <run>/overhead_report.json.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import d1_scores_report as scores_report
import numpy as np

import hulder

CHANGES = ("prefix_caching", "gpu", "batch_size")


def caching_outcome(verdicts: list[str]) -> str:
    if not verdicts:
        return "not measured"
    return "holds at this threshold on this set" if set(verdicts) == {"PASS"} else "needs its own calibration"


def _subset(snapshot: hulder.Snapshot, keep: np.ndarray, metadata: dict | None = None) -> hulder.Snapshot:
    """Prompt id hashes describe the full set, so they go when rows do."""
    drop = {"prompt_ids_sha256", "canary_sha256", "gpu_uuid", "gpu_driver"}
    return hulder.Snapshot(
        ids=snapshot.ids[keep],
        labels=snapshot.labels[keep],
        scores=snapshot.scores[keep],
        metadata={**{k: v for k, v in snapshot.metadata.items() if k not in drop}, **(metadata or {})},
    )


def stage_report(
    reference: hulder.Snapshot, rerun: hulder.Snapshot, stage: dict, threshold: float, target_fpr: float
) -> dict:
    """One stage through check_snapshots, on the rows that have a score, in the reference's order.
    reference and rerun hold scores. stage has ids, scores and prefix_caching, and any other metadata."""
    row = {str(i): k for k, i in enumerate(stage["ids"])}
    aligned = np.asarray(stage["scores"])[[row[str(i)] for i in reference.ids]]
    keep = ~np.isnan(aligned)
    metadata = {k: v for k, v in stage.items() if k not in ("ids", "scores")}
    candidate = hulder.Snapshot(
        ids=reference.ids[keep],
        labels=reference.labels[keep],
        scores=aligned[keep],
        metadata={**_subset(reference, keep).metadata, **metadata},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = hulder.check_snapshots(
            _subset(reference, keep),
            candidate,
            _subset(rerun, keep),
            threshold=threshold,
            target_fpr=target_fpr,
            changes_under_test=CHANGES,
        )
    return {
        "verdict": result.verdict.value,
        "rows": int(keep.sum()),
        "without_score": int((~keep).sum()),
        "decision_flips": int(result.decision_flips),
        "mean_abs_shift": float(result.mean_abs_shift),
        "noise_floor_mean_abs_shift": float(result.noise_floor_mean_abs_shift),
        "noise_floor_decision_flips": int(result.noise_floor_decision_flips),
        "shift_exceeds_noise_floor": result.shift_exceeds_noise_floor,
        "changes": [list(change) for change in result.changes_under_test],
        "warnings": [str(w.message) for w in caught],
    }


def scored(snapshot: hulder.Snapshot, probe: hulder.Probe) -> hulder.Snapshot:
    return hulder.Snapshot(
        ids=snapshot.ids,
        labels=snapshot.labels,
        scores=hulder.apply_probe(probe, snapshot.activations),
        metadata={**snapshot.metadata, "prefix_caching": False},
    )


def report(run_dir: Path, reference: hulder.Snapshot, rerun: hulder.Snapshot, probe_path: Path) -> dict:
    results = json.loads((run_dir / "d1_l4.json").read_text())
    if results["stopped"]:
        return {"stopped": results["stopped"]}
    probe = hulder.Probe.load(probe_path)
    threshold, target_fpr = probe.metadata["threshold"], probe.metadata["target_fpr"]
    reference_scores, rerun_scores = scored(reference, probe), scored(rerun, probe)
    stages = {}
    for run in results["runs"]:
        if "score" not in run:
            continue
        name = f"r{run['round']}_{run['arm']}"
        with np.load(run_dir / name / "score.npz", allow_pickle=False) as saved:
            stage = {
                "ids": [str(i) for i in saved["ids"]],
                "scores": saved["scores"],
                "prefix_caching": run["arm"] == "scoring_caching",
                "batch_size": run["score"]["concurrency"],
            }
        stages[name] = {
            **stage_report(reference_scores, rerun_scores, stage, threshold, target_fpr),
            "arm": run["arm"],
            "round": run["round"],
            "cache": run["score"].get("cache"),
        }
    caching = [s["verdict"] for s in stages.values() if s["arm"] == "scoring_caching"]
    return {
        "stopped": None,
        "threshold": threshold,
        "target_fpr": target_fpr,
        "stages": stages,
        "caching_outcome": caching_outcome(caching),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", choices=("qwen3", "1.5b"), required=True)
    args = parser.parse_args(argv)
    inputs = scores_report.frozen_inputs() if args.model == "qwen3" else scores_report.smoke_inputs()
    reference, rerun, probe_path, _ = inputs
    result = report(args.run_dir, reference, rerun, probe_path)
    (args.run_dir / "overhead_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[:6000])
    return 0 if result["stopped"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
