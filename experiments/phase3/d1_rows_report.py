"""D1 confirmation: which rows of each score stage moved, and whether a server changed path.

    source scripts/dev-env.sh && uv run python experiments/phase3/d1_rows_report.py .cache/modal-runs/<run id>

Reads a finished run's score stages (d1_scores_report.py scores them) and separates rows bit-identical
to kept/, which shift by exactly 0, from those that changed. So a stage's mean |Δscore| can be read as
how many rows changed and how far. Also counts rows bit-identical between every pair of captures, and
puts each server's greedy divergences beside them, since those come before the score stage and its
arrival order. Prints JSON to stdout and scores nothing against a rule. CPU only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_scores_report as scores

import hulder

ROUNDS = (1, 2, 3)
ARMS = (scores.STOCK, scores.ZERO_DRAFT)


def row_shift(probe: hulder.Probe, reference: np.ndarray, other: np.ndarray) -> dict:
    """other's shift from the reference, over all rows and over the rows that aren't bit-identical."""
    a, b = np.asarray(reference), np.asarray(other)
    same = (a == b).all(axis=1)
    moved = np.abs(hulder.apply_probe(probe, b) - hulder.apply_probe(probe, a))
    wide_a, wide_b = a.astype(np.float32), b.astype(np.float32)
    relative_l2 = np.linalg.norm(wide_b - wide_a, axis=1) / np.linalg.norm(wide_a, axis=1)
    changed = ~same
    return {
        "rows_bit_identical": int(same.sum()),
        "mean_abs_dscore": float(moved.mean()),
        "changed_mean_abs_dscore": float(moved[changed].mean()) if changed.any() else 0.0,
        "changed_median_abs_dscore": float(np.median(moved[changed])) if changed.any() else 0.0,
        "changed_mean_relative_l2": float(relative_l2[changed].mean()) if changed.any() else 0.0,
    }


def identical_rows(captures: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Rows bit-identical between every pair of captures on the same prompts in the same order."""
    return {a: {b: int((captures[a] == captures[b]).all(axis=1).sum()) for b in captures} for a in captures}


def in_reference_order(run_dir: Path, name: str, reference_ids: list[str]) -> np.ndarray:
    with np.load(run_dir / name / "score.npz", allow_pickle=False) as saved:
        row = {str(i): k for k, i in enumerate(saved["ids"])}
        return saved["activations"][[row[i] for i in reference_ids]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    reference, rerun, probe_path, _ = scores.frozen_inputs()
    probe = hulder.Probe.load(probe_path)
    ids = [str(i) for i in reference.ids]
    kept, shuffled = np.asarray(reference.activations), np.asarray(rerun.activations)
    stages = {f"r{rnd}_{arm}": in_reference_order(args.run_dir, f"r{rnd}_{arm}", ids) for rnd in ROUNDS for arm in ARMS}
    results = json.loads((args.run_dir / "d1_l4.json").read_text())
    out = {
        "shift_from_kept": {"shuffled": row_shift(probe, kept, shuffled)}
        | {name: row_shift(probe, kept, rows) for name, rows in stages.items()},
        "identical_rows": identical_rows({"kept": kept, "shuffled": shuffled, **stages}),
        "greedy_first_divergences": {
            name: [[d["prompt"], d["token"]] for d in run["diverged"]] for name, run in results["generation"].items()
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
