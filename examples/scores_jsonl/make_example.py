"""Synthetic scores for the bring-your-own-scores example: a reference, a rerun of it, and a
candidate whose scores rose, as JSONL (docs/design.md §4). The scores are made up, from a fixed
seed, so the example needs no model and no GPU.

    python examples/scores_jsonl/make_example.py

Writes reference.jsonl, rerun.jsonl and candidate.jsonl next to this file and prints the
threshold, fitted on the reference's benign scores for a 5% target FPR.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 0
N_BENIGN, N_ATTACK = 400, 300
TARGET_FPR = 0.05
# The candidate's logits rise by this much, which raises its FPR at the old threshold to about a third.
CANDIDATE_SHIFT = 1.2
RERUN_NOISE = 0.005  # logit noise between two runs of the reference


def scores(seed: int = SEED) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(N_BENIGN, int), np.ones(N_ATTACK, int)]
    logits = np.r_[rng.normal(-3.5, 1, N_BENIGN), rng.normal(0, 1, N_ATTACK)]
    order = rng.permutation(len(labels))  # benign and attack interleaved, as real data would be
    labels, logits = labels[order], logits[order]
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    runs = {
        "reference": sigmoid(logits),
        "rerun": sigmoid(logits + rng.normal(0, RERUN_NOISE, len(logits))),
        "candidate": sigmoid(logits + CANDIDATE_SHIFT),
    }
    return [f"q{i:04d}" for i in range(len(labels))], labels, runs


def threshold(reference: np.ndarray, labels: np.ndarray) -> float:
    """The reference's benign 95th percentile, flagging scores strictly above it."""
    return float(np.quantile(reference[labels == 0], 1 - TARGET_FPR, method="higher"))


def write(out: Path = HERE) -> float:
    ids, labels, runs = scores()
    metadata = {"model": "example-scorer", "base_model": "example-scorer"}
    for role, values in runs.items():
        meta = {**metadata, "quantization": "int8"} if role == "candidate" else metadata
        lines = [json.dumps({"metadata": meta})]
        lines += [
            json.dumps({"id": i, "score": round(float(s), 6), "label": int(lab)})
            for i, s, lab in zip(ids, values, labels, strict=True)
        ]
        (out / f"{role}.jsonl").write_text("\n".join(lines) + "\n")
    return threshold(np.round(runs["reference"], 6), labels)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    print(f"threshold {write(out)!r} at a {TARGET_FPR:.0%} target FPR")
