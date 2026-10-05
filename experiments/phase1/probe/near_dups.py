"""Does near-duplication between train and calibration explain the calibration-vs-test FNR gap?

Two levels of duplicate, both on normalised text (lowercase, punctuation removed,
whitespace collapsed):
  normalised   exact match of the normalised text
  near         cosine >= SIM of character 3-5-gram TF-IDF vectors to the nearest train item

For calibration, validation and test it reports the share of items with a near-duplicate
in train, and the main probe's FNR and FPR split by that flag. If calibration's FNR on items
without a near-duplicate matches test's, near-duplicates explain the gap.

Run from the repo root, after train.py:
    source scripts/dev-env.sh && uv run python experiments/phase1/probe/near_dups.py

Writes docs/results/phase1/near_dups.json.
"""

import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import hulder

SPLITS = Path(".cache/probe/splits.parquet")
ACTS = Path(".cache/probe/fp16")
PROBE = Path(".cache/probe/probe_main.npz")
OUT = Path("docs/results/phase1/near_dups.json")
SIMS = (0.8, 0.9)
EVAL = ("calibration", "validation", "test")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCT)).strip()


def max_sim_to(train_vecs, vecs, chunk: int = 1024) -> np.ndarray:
    out = []
    for i in range(0, vecs.shape[0], chunk):
        out.append((vecs[i : i + chunk] @ train_vecs.T).max(axis=1).toarray().ravel())
    return np.concatenate(out)


def rates(flagged: np.ndarray, y: np.ndarray) -> dict:
    return {
        "fnr": float((~flagged & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "n_attack": int((y == 1).sum()),
        "fpr": float((flagged & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "n_benign": int((y == 0).sum()),
    }


def main() -> None:
    df = pd.read_parquet(SPLITS)
    df = df[~df["split"].str.startswith("bpairs_")].copy()
    df["norm"] = df["text"].map(normalise)
    split = df["split"].to_numpy()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    vec.fit(df.loc[split == "train", "norm"])
    tf = vec.transform(df["norm"])  # rows are L2-normalised
    train_tf = tf[split == "train"]
    train_norm = set(df.loc[split == "train", "norm"])

    probe = hulder.Probe.load(PROBE)
    acts = np.load(ACTS / "activations.npy", mmap_mode="r")
    layers = json.loads((ACTS / "manifest.json").read_text())["capture_config"]["layers"]
    li = layers.index(probe.metadata["layer"])
    all_ids = json.loads((ACTS / "ids.json").read_text())
    row_of = {i: r for r, i in enumerate(all_ids)}

    report: dict = {"normalise": "lowercase, strip punctuation, collapse whitespace", "tfidf": "char_wb 3-5, min_df 2, fit on train"}
    for s in EVAL:
        m = split == s
        sub = df[m]
        y = sub["label"].to_numpy()
        sim = max_sim_to(train_tf, tf[m])
        rows = np.array([row_of[i] for i in sub["id"]])
        flagged = hulder.apply_probe(probe, np.asarray(acts[rows, li, 0])) > probe.metadata["threshold"]
        entry = {
            "n": int(m.sum()),
            "normalised_dup_in_train": int(sub["norm"].isin(train_norm).sum()),
            "max_sim_to_train_median": {lab: float(np.median(sim[y == k])) for k, lab in ((0, "benign"), (1, "attack"))},
            "all": rates(flagged, y),
        }
        for t in SIMS:
            near = sim >= t
            entry[f"near_dup_in_train@{t}"] = {
                "share": float(near.mean()),
                "share_benign": float(near[y == 0].mean()),
                "share_attack": float(near[y == 1].mean()),
                "with_near_dup": rates(flagged[near], y[near]),
                "without_near_dup": rates(flagged[~near], y[~near]),
            }
        report[s] = entry
        print(s, json.dumps(entry, indent=1))

    # Cross-split near-dup rates among the evaluation splits themselves.
    report["cross_eval_near_dup@0.9"] = {}
    for a, b in (("calibration", "test"), ("validation", "test"), ("calibration", "validation")):
        sim = max_sim_to(tf[split == b], tf[split == a])
        report["cross_eval_near_dup@0.9"][f"{a}->{b}"] = float((sim >= 0.9).mean())
    print(json.dumps(report["cross_eval_near_dup@0.9"], indent=1))
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
