"""Train the Qwen3-4B probe on the full capture's kept/ (docs/results/phase4/full_capture.md).

Six cells, layers 18 and 27 at positions -1, -9 and -10, each a logistic regression on the
Phase 1 splits (docs/design.md §7), fitted as experiments/phase1/probe/train.py fits:
  train        fits weights and the standardisation mean/std
  validation   picks C per cell, and the primary cell at position -1 (docs/design.md §5 D2)
  calibration  sets each cell's threshold for 1% FPR on benign scores
  test, bpairs scored only by the score stage, after selection.json is frozen and committed

C starts on a grid of decades 1e-5 to 1. Where a cell's strict validation maximum is on an edge
of its grid, the grid extends one decade past that edge, at most 3 times; a cell still on an
edge is frozen as it stands and flagged. A tie within 1e-4 of the maximum goes to the smaller C.

Run from the repo root. The select stage refuses to overwrite a frozen selection, and the score
stage runs only on the committed selection (its sha256) and only once:
    source scripts/dev-env.sh && uv run python experiments/phase4/probe/train_qwen3.py select
    source scripts/dev-env.sh && uv run python experiments/phase4/probe/train_qwen3.py score

select writes docs/results/phase4/probe_qwen3_selection.json and each cell's probe under
.cache/probe/qwen3/; score writes docs/results/phase4/probe_qwen3.json.
"""

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import hulder

KEPT = Path(".cache/modal-runs/20261022T125644Z/kept")
SPLITS = Path(".cache/probe/splits.parquet")
SELECTION = Path("docs/results/phase4/probe_qwen3_selection.json")
PROBES = Path(".cache/probe/qwen3")
CELLS = tuple((layer, position) for layer in (18, 27) for position in (-1, -9, -10))
GRID = tuple(range(-5, 1))  # C = 10**e
TIE = 1e-4
MAX_EXTENSIONS = 3
TARGET_FPR = 0.01
SELECT_SPLITS = ("train", "validation", "calibration")
SPLIT_NAMES = (*SELECT_SPLITS, "test", "bpairs")  # bpairs: all 1,200 boundary-pairs, every bpairs_* split
SCORES = Path("docs/results/phase4/probe_qwen3.json")
# The selection reviewed on 2026-10-23 and committed in d0d0b0bf81f8.
SELECTION_SHA256 = "fbc7306b83b176e344880a300f42f38b5ab2a21ecb4e77c03b0bcdb04a45140c"
QWEN25_BPAIRS_AUROC = 0.839  # docs/results/phase1/probe_main.md
N_BOOT = 2000
SEED = 0

Rows = Callable[[str, int, int], tuple[np.ndarray, np.ndarray]]


def c_of(exponent: int) -> float:
    return float(f"1e{exponent}")


def fit(x: np.ndarray, y: np.ndarray, c: float, meta: dict) -> hulder.Probe:
    x = x.astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=c, max_iter=5000).fit((x - mean) / std, y)
    return hulder.Probe(weights=clf.coef_[0], bias=float(clf.intercept_[0]), mean=mean, std=std, metadata=meta)


def threshold_for_fpr(benign_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(benign_scores, 1 - fpr, method="higher"))


def choose_c(aurocs: dict[int, float]) -> int:
    """The smallest C whose validation AUROC is within TIE of the maximum."""
    best = max(aurocs.values())
    return min(e for e, a in aurocs.items() if a >= best - TIE)


def _edge(exponents: list[int], e: int) -> str:
    return "lower edge" if e == min(exponents) else "upper edge" if e == max(exponents) else "interior"


def _strict_edge(aurocs: dict[int, float]) -> str:
    """The edge whose C beats every other C outright, or "interior". An exact tie isn't a maximum on an
    edge, so a flat grid doesn't extend."""
    best = max(aurocs.values())
    top = [e for e, a in aurocs.items() if a == best]
    return _edge(sorted(aurocs), top[0]) if len(top) == 1 else "interior"


def select_c(val_auroc: Callable[[float], float], grid: tuple[int, ...] = GRID) -> dict:
    """C for one cell, extending the grid past an edge that holds the strict maximum."""
    aurocs = {e: val_auroc(c_of(e)) for e in grid}
    extensions = []
    side = _strict_edge(aurocs)
    while side != "interior" and len(extensions) < MAX_EXTENSIONS:
        e = min(aurocs) - 1 if side == "lower edge" else max(aurocs) + 1
        aurocs[e] = val_auroc(c_of(e))
        extensions.append(c_of(e))
        side = _strict_edge(aurocs)
    chosen = choose_c(aurocs)
    return {
        "C": c_of(chosen),
        "val_auroc": aurocs[chosen],
        "grid": [{"C": c_of(e), "val_auroc": aurocs[e]} for e in sorted(aurocs)],
        "position_in_grid": _edge(sorted(aurocs), chosen),
        "extensions": extensions,
        "edge_unresolved": side != "interior",
    }


def choose_primary(cells: dict[tuple[int, int], dict]) -> tuple[int, int]:
    """The best validation AUROC at position -1, a tie within TIE going to the smaller layer."""
    at_last = {cell: c for cell, c in cells.items() if cell[1] == -1}
    best = max(c["val_auroc"] for c in at_last.values())
    return min(cell for cell, c in at_last.items() if c["val_auroc"] >= best - TIE)


def probe_metadata(capture_meta: dict, *, layer: int, position: int, c: float) -> dict:
    """What the probe records: the capture's model and prompt settings, so docs/design.md §6
    metadata rule 8 holds against kept/, plus its own cell and training settings."""
    keys = ("model", "revision", "base_model", "system_prompt", "dtype", "enable_thinking")
    return {
        **{k: capture_meta[k] for k in keys if k in capture_meta},
        "layer": layer,
        "position": position,
        "C": c,
        "task": "prompt attack (injection or jailbreak)",
        "training_data": "S-Labs train split after dedup (docs/results/phase1/probe_splits.json)",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_stage(rows: Rows, out: Path, *, probe_dir: Path, source: dict, capture_meta: dict | None = None) -> dict:
    """Fit every cell, choose C and the primary cell on validation, set thresholds on calibration,
    save each cell's probe and freeze the selection to out. Reads only SELECT_SPLITS."""
    if out.exists():
        raise SystemExit(f"{out} is frozen: a new selection is a new, labelled result, not an overwrite")
    cells, probes = {}, {}
    for layer, position in CELLS:
        x_train, y_train = rows("train", layer, position)
        x_val, y_val = rows("validation", layer, position)

        def val_auroc(c, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val):
            return float(roc_auc_score(y_val, hulder.apply_probe(fit(x_train, y_train, c, {}), x_val)))

        cell = select_c(val_auroc)
        meta = probe_metadata(capture_meta or {}, layer=layer, position=position, c=cell["C"])
        probe = fit(x_train, y_train, cell["C"], meta)
        x_cal, y_cal = rows("calibration", layer, position)
        threshold = threshold_for_fpr(hulder.apply_probe(probe, x_cal)[y_cal == 0], TARGET_FPR)
        probes[(layer, position)] = hulder.Probe(
            weights=probe.weights,
            bias=probe.bias,
            mean=probe.mean,
            std=probe.std,
            metadata={**meta, "threshold": threshold, "target_fpr": TARGET_FPR},
        )
        cells[(layer, position)] = {**cell, "threshold": threshold}
        print(
            f"layer {layer} position {position:>3}: C {cell['C']:g} ({cell['position_in_grid']}), "
            f"val AUROC {cell['val_auroc']:.5f}",
            flush=True,
        )
    primary = choose_primary(cells)
    # Fitting is deterministic (lbfgs, no subsampling): the primary cell refitted must match bit for bit.
    x_train, y_train = rows("train", *primary)
    again = fit(x_train, y_train, cells[primary]["C"], {})
    refit_identical = (
        again.weights.tobytes() == probes[primary].weights.tobytes() and again.bias == probes[primary].bias
    )
    probe_dir.mkdir(parents=True, exist_ok=True)
    named = {}
    for (layer, position), cell in cells.items():
        path = probe_dir / f"{layer}_{position}.npz"
        probes[(layer, position)].save(path)
        named[f"{layer}_{position}"] = {**cell, "probe_sha256": _sha256(path)}
    selection = {
        "primary": f"{primary[0]}_{primary[1]}",
        "cells": named,
        "rules": {
            "grid": [c_of(e) for e in GRID],
            "tie": TIE,
            "max_extensions": MAX_EXTENSIONS,
            "target_fpr": TARGET_FPR,
            "threshold": "benign calibration quantile 1 - target_fpr, method='higher'",
        },
        "refit_identical": refit_identical,
        "source": source,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def pair_of(prompt_id: str) -> str:
    """A boundary-pairs id's pair, apibp_0059 for bpairs/apibp_0059_a and _b. Grouping by it gives the
    dataset's own 600 pair_id groups, checked on 2026-10-23."""
    return prompt_id.split("/", 1)[1].rsplit("_", 1)[0]


def rate(k: int, n: int) -> dict:
    return {"rate": k / n if n else None, "k": k, "n": n, "ci95": list(hulder.clopper_pearson(k, n)) if n else None}


def at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    flagged = scores > threshold
    return {
        "fpr": rate(int((flagged & (y == 0)).sum()), int((y == 0).sum())),
        "fnr": rate(int((~flagged & (y == 1)).sum()), int((y == 1).sum())),
    }


def auroc_ci(scores: np.ndarray, y: np.ndarray, clusters: list[str] | None = None) -> dict:
    """AUROC with a percentile bootstrap, resampling whole clusters (boundary-pairs pairs) when given."""
    rng = np.random.default_rng(SEED)
    groups = np.arange(len(y)) if clusters is None else np.asarray(clusters)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], scores[idx]))
    return {"auroc": float(roc_auc_score(y, scores)), "ci95": np.percentile(boots, [2.5, 97.5]).tolist()}


def score_stage(
    rows: Rows,
    ids_of: Callable[[str], list[str]],
    *,
    selection: Path,
    probe_dir: Path,
    out: Path,
    source: dict,
    expected_sha256: str = SELECTION_SHA256,
) -> dict:
    """Score calibration, test and all of boundary-pairs once for every cell, at the frozen thresholds.
    Runs only on the committed selection and its probes, and only once."""
    got = _sha256(selection) if selection.exists() else None
    if got != expected_sha256:
        raise SystemExit(f"{selection}'s sha256 is {got}, not the committed selection's {expected_sha256}")
    if out.exists():
        raise SystemExit(f"{out} exists: test is already scored, and is scored once")
    frozen = json.loads(selection.read_text())
    for name, cell in frozen["cells"].items():
        if _sha256(probe_dir / f"{name}.npz") != cell["probe_sha256"]:
            raise SystemExit(f"{probe_dir / name}.npz isn't the probe the selection froze for cell {name}")
    pairs = [pair_of(i) for i in ids_of("bpairs")]
    cells = {}
    for name, cell in frozen["cells"].items():
        layer, position = (int(v) for v in name.split("_"))
        probe = hulder.Probe.load(probe_dir / f"{name}.npz")
        threshold = cell["threshold"]
        scored = {}
        for split in ("calibration", "test", "bpairs"):
            x, y = rows(split, layer, position)
            scored[split] = (hulder.apply_probe(probe, x), y)
        cal, test, bp = scored["calibration"], scored["test"], scored["bpairs"]
        cells[name] = {
            "threshold": threshold,
            "calibration": {
                **at_threshold(*cal, threshold),
                "median_benign_score": float(np.median(cal[0][cal[1] == 0])),
            },
            "test": {**auroc_ci(*test), **at_threshold(*test, threshold)},
            "bpairs": {
                **auroc_ci(*bp, clusters=pairs),
                **at_threshold(*bp, threshold),
                "bootstrap": "pair-clustered",
                "median_benign_score": float(np.median(bp[0][bp[1] == 0])),
                "qwen25_auroc": QWEN25_BPAIRS_AUROC,
            },
        }
        print(
            f"{name}: test AUROC {cells[name]['test']['auroc']:.4f}, bpairs AUROC {cells[name]['bpairs']['auroc']:.4f}"
        )
    result = {
        "primary": frozen["primary"],
        "selection_sha256": got,
        "cells": cells,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "source": source,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def kept_rows() -> tuple[Rows, Callable[[str], list[str]], dict, dict]:
    """A row reader over kept/ by split, each split's ids, the capture's metadata, and the inputs'
    provenance. One array is in memory at a time, and the splits' labels must equal kept/'s."""
    import pandas as pd

    df = pd.read_parquet(SPLITS, columns=["id", "split", "label"])
    split_of = dict(zip(df["id"], df["split"], strict=True))
    label_of = dict(zip(df["id"], df["label"], strict=True))
    cache: dict[str, hulder.Snapshot] = {}

    def snapshot(layer: int, position: int) -> hulder.Snapshot:
        name = f"{layer}_{position}"
        if name not in cache:
            cache.clear()  # one array in memory at a time
            cache[name] = hulder.Snapshot.load(KEPT / f"{name}.npz")
        return cache[name]

    first = snapshot(*CELLS[0])
    ids = [str(i) for i in first.ids]
    if [int(label_of[i]) for i in ids] != [int(v) for v in first.labels]:
        raise ValueError("kept/'s labels differ from splits.parquet's")
    group = {s: s for s in SELECT_SPLITS + ("test",)} | {
        s: "bpairs" for s in set(split_of.values()) if s.startswith("bpairs_")
    }
    rows_of = {s: np.array([i for i, x in enumerate(ids) if group[split_of[x]] == s]) for s in SPLIT_NAMES}

    def rows(split: str, layer: int, position: int) -> tuple[np.ndarray, np.ndarray]:
        s = snapshot(layer, position)
        if [str(i) for i in s.ids] != ids:
            raise ValueError(f"kept/{layer}_{position}.npz holds other prompts than the other arrays")
        r = rows_of[split]
        return np.asarray(s.activations[r]), np.asarray(s.labels[r])

    sums = (KEPT / "SHA256SUMS").read_text() if (KEPT / "SHA256SUMS").exists() else None
    source = {
        "kept": str(KEPT),
        "kept_prompt_ids_sha256": first.metadata.get("prompt_ids_sha256"),
        "splits_sha256": _sha256(SPLITS),
        "kept_sha256sums": sums,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "packages": {p: version(p) for p in ("hulder", "numpy", "scikit-learn", "pandas")},
        "counts": {s: len(r) for s, r in rows_of.items()},
    }
    return rows, lambda split: [ids[i] for i in rows_of[split]], dict(first.metadata), source


def main(stage: str) -> None:
    rows, ids_of, meta, source = kept_rows()
    if stage == "select":
        selection = select_stage(rows, SELECTION, probe_dir=PROBES, source=source, capture_meta=meta)
        print(json.dumps({k: v for k, v in selection.items() if k != "source"}, indent=2))
    elif stage == "score":
        score_stage(rows, ids_of, selection=SELECTION, probe_dir=PROBES, out=SCORES, source=source)
    else:
        raise SystemExit("usage: train_qwen3.py select|score")


if __name__ == "__main__":
    main(*sys.argv[1:])
