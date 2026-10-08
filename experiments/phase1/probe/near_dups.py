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
from common import ACTS, SPLITS, Activations, run_manifest
from sklearn.feature_extraction.text import TfidfVectorizer
from train import at_threshold

import hulder

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


def main() -> None:
    df = pd.read_parquet(SPLITS)
    df = df[~df["split"].str.startswith("bpairs_")].copy()
    df["norm"] = df["text"].map(normalise)
    split = df["split"].to_numpy()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    vec.fit(df.loc[split == "train", "norm"])
    tfidf = vec.transform(df["norm"])  # rows are L2-normalised
    train_tfidf = tfidf[split == "train"]
    train_norm = set(df.loc[split == "train", "norm"])

    probe = hulder.Probe.load(PROBE)
    acts = Activations.load(ACTS)

    report: dict = {
        "normalise": "lowercase, strip punctuation, collapse whitespace",
        "tfidf": "char_wb 3-5, min_df 2, fit on train",
    }
    for s in EVAL:
        m = split == s
        sub = df[m]
        y = sub["label"].to_numpy()
        sim = max_sim_to(train_tfidf, tfidf[m])
        scores = hulder.apply_probe(probe, acts.for_probe(sub["id"], probe))
        threshold = probe.metadata["threshold"]
        entry = {
            "n": int(m.sum()),
            "normalised_dup_in_train": int(sub["norm"].isin(train_norm).sum()),
            "max_sim_to_train_median": {
                lab: float(np.median(sim[y == k])) for k, lab in ((0, "benign"), (1, "attack"))
            },
            "all": at_threshold(scores, y, threshold),
        }
        for sim_cut in SIMS:
            near = sim >= sim_cut
            entry[f"near_dup_in_train@{sim_cut}"] = {
                "share": float(near.mean()),
                "share_benign": float(near[y == 0].mean()),
                "share_attack": float(near[y == 1].mean()),
                "with_near_dup": at_threshold(scores[near], y[near], threshold),
                "without_near_dup": at_threshold(scores[~near], y[~near], threshold),
            }
        report[s] = entry
        print(s, json.dumps(entry, indent=1))

    # Cross-split near-dup rates among the evaluation splits themselves.
    report["cross_eval_near_dup@0.9"] = {}
    for a, b in (("calibration", "test"), ("validation", "test"), ("calibration", "validation")):
        sim = max_sim_to(tfidf[split == b], tfidf[split == a])
        report["cross_eval_near_dup@0.9"][f"{a}->{b}"] = float((sim >= 0.9).mean())
    print(json.dumps(report["cross_eval_near_dup@0.9"], indent=1))
    report["run_manifest"] = run_manifest(inputs=(SPLITS, PROBE, ACTS / "activations.npy"))
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
