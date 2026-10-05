"""Build the probe's four S-Labs splits plus the held-out boundary-pairs set.

Steps, in order:
  1. Normalise text (strip, collapse whitespace) to a dedup key.
  2. Within each S-Labs split, drop every copy of a key with conflicting labels, then keep
     the first copy of each remaining key.
  3. Across splits, a key keeps its copy in the highest-priority split
     (test > validation > train), so the official eval splits stay as intact as possible.
     A key whose labels disagree across splits is dropped from every split.
  4. Drop from S-Labs any key that also appears in boundary-pairs, which is never trained on.
  5. Carve a stratified calibration split of 2,101 out of train (seed 0).
  6. Report pairwise overlaps across all four splits after the fact (all must be 0).

Run from the repo root:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/prepare_splits.py

Writes .cache/probe/splits.parquet (regenerable) and docs/results/phase1/probe_splits.json.
"""

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SLABS = ("S-Labs/prompt-injection-dataset", "002a9dd18514abd021869823d6b0429b38606d99")
BPAIRS = ("3nesdeniz/agentic-prompt-injection-boundary-pairs", "a5682e7573e1c7bc4b12e64d49c0dcd90ca776cf")
CALIBRATION_SIZE = 2101  # same size as S-Labs' validation and test splits
SEED = 0
OUT = Path(".cache/probe/splits.parquet")
REPORT = Path("docs/results/phase1/probe_splits.json")


def key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_slabs(split: str) -> pd.DataFrame:
    path = hf_hub_download(SLABS[0], f"data/{split}.csv", repo_type="dataset", revision=SLABS[1])
    df = pd.read_csv(path)
    df["id"] = [f"slabs/{split}/{i}" for i in range(len(df))]
    df["split"] = split
    return df


def dedup_within(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    labels_per_key = df.groupby("key")["label"].nunique()
    conflicted = set(labels_per_key[labels_per_key > 1].index)
    kept = df[~df["key"].isin(conflicted)].drop_duplicates("key", keep="first")
    stats = {
        "rows": len(df),
        "label_conflict_keys": len(conflicted),
        "rows_dropped_conflict": int(df["key"].isin(conflicted).sum()),
        "rows_dropped_duplicate": len(df) - int(df["key"].isin(conflicted).sum()) - len(kept),
        "rows_kept": len(kept),
    }
    return kept, stats


def main() -> None:
    report: dict = {"slabs_revision": SLABS[1], "bpairs_revision": BPAIRS[1], "dedup_key": "strip + collapse whitespace"}

    splits = {}
    report["within_split"] = {}
    for s in ["train", "validation", "test"]:
        df = load_slabs(s)
        df["key"] = df["text"].map(key)
        splits[s], report["within_split"][s] = dedup_within(df)

    # Across splits: conflicting labels drop everywhere; otherwise higher priority wins.
    allrows = pd.concat(splits.values())
    labels_per_key = allrows.groupby("key")["label"].nunique()
    cross_conflict = set(labels_per_key[labels_per_key > 1].index)
    report["cross_split"] = {"label_conflict_keys": len(cross_conflict), "removed": {}}
    priority = ["test", "validation", "train"]
    claimed: set[str] = set()
    for s in priority:
        df = splits[s]
        drop = df["key"].isin(cross_conflict) | df["key"].isin(claimed)
        report["cross_split"]["removed"][s] = {
            "label_conflict": int(df["key"].isin(cross_conflict).sum()),
            "duplicate_of_higher_priority_split": int((df["key"].isin(claimed) & ~df["key"].isin(cross_conflict)).sum()),
        }
        splits[s] = df[~drop]
        claimed |= set(splits[s]["key"])

    # Boundary-pairs is held out from the main probe, so its texts leave S-Labs.
    bp = []
    for s in ["train", "validation", "test"]:
        path = hf_hub_download(BPAIRS[0], f"data/{s}.parquet", repo_type="dataset", revision=BPAIRS[1])
        bp.append(pd.read_parquet(path)[["id", "text", "label"]].assign(split=f"bpairs_{s}"))
    bp = pd.concat(bp)
    bp["key"] = bp["text"].map(key)
    bp["id"] = "bpairs/" + bp["id"]
    report["removed_overlap_with_bpairs"] = {}
    for s in priority:
        hit = splits[s]["key"].isin(set(bp["key"]))
        report["removed_overlap_with_bpairs"][s] = int(hit.sum())
        splits[s] = splits[s][~hit]

    # Stratified calibration split carved from train.
    train = splits["train"]
    rng = np.random.default_rng(SEED)
    frac = CALIBRATION_SIZE / len(train)
    calib_idx = []
    for _, group in train.groupby("label"):
        n = round(len(group) * frac)
        calib_idx.extend(rng.choice(group.index.to_numpy(), size=n, replace=False))
    # Labels are integer-indexed per original split; index values are unique within train.
    is_calib = train.index.isin(calib_idx)
    splits["calibration"] = train[is_calib].assign(split="calibration")
    splits["train"] = train[~is_calib]

    order = ["train", "calibration", "validation", "test"]
    report["pairwise_overlap_after"] = {
        f"{a}&{b}": len(set(splits[a]["key"]) & set(splits[b]["key"])) for a, b in itertools.combinations(order, 2)
    }
    assert all(v == 0 for v in report["pairwise_overlap_after"].values())
    report["final"] = {
        s: {"n": len(splits[s]), "benign": int((splits[s]["label"] == 0).sum()), "attack": int((splits[s]["label"] == 1).sum())}
        for s in order
    }
    report["final"]["bpairs_all"] = {"n": len(bp), "benign": int((bp["label"] == 0).sum()), "attack": int((bp["label"] == 1).sum())}

    out = pd.concat([splits[s] for s in order] + [bp])[["id", "split", "text", "label"]].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
